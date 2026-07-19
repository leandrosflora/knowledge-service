from __future__ import annotations

import re
import time
import uuid
from contextvars import ContextVar
from typing import Any, Iterable

import jwt
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

TENANT_CLAIM = "tenant_id"
_tenant_id: ContextVar[str | None] = ContextVar("tenant_id", default=None)

HTTP_REQUESTS = Counter(
    "platform_http_requests_total",
    "Total HTTP requests handled by the service.",
    ["service", "method", "path", "status"],
)
HTTP_DURATION = Histogram(
    "platform_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["service", "method", "path"],
)
AUTH_FAILURES = Counter(
    "platform_internal_auth_failures_total",
    "Rejected internal authentication attempts.",
    ["service", "reason"],
)


def normalize_tenant_id(value: str | None) -> str:
    try:
        parsed = uuid.UUID((value or "").strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError("Tenant ID must be a UUID") from exc
    if parsed.int == 0:
        raise ValueError("Tenant ID cannot be empty UUID")
    return str(parsed)


def current_tenant_id() -> str:
    tenant_id = _tenant_id.get()
    if not tenant_id:
        raise RuntimeError("Tenant context is not available")
    return tenant_id


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class PlatformMiddleware:
    def __init__(
        self,
        app,
        *,
        settings: Any,
        public_paths: Iterable[str] = (),
        tenant_required_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.settings = settings
        self.public_paths = tuple(public_paths)
        self.tenant_required_paths = tuple(tenant_required_paths)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "UNKNOWN")
        normalized_path = _normalize_path(path)
        started = time.perf_counter()
        status_code = 500
        tenant_token = None

        async def capture_status(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            headers = {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in scope.get("headers", [])
            }
            claims: dict[str, Any] | None = None
            if not _matches(path, self.public_paths):
                auth_result = self._authenticate(headers.get("authorization"))
                if isinstance(auth_result, JSONResponse):
                    status_code = auth_result.status_code
                    await auth_result(scope, receive, send)
                    return
                claims = auth_result

            if _matches(path, self.tenant_required_paths):
                try:
                    header_tenant = normalize_tenant_id(headers.get("x-tenant-id"))
                except ValueError:
                    status_code = 400
                    await JSONResponse(
                        {"detail": "X-Tenant-Id must be a non-empty UUID."},
                        status_code=400,
                    )(scope, receive, send)
                    return

                if self.settings.internal_auth_enabled:
                    try:
                        claim_tenant = normalize_tenant_id((claims or {}).get(TENANT_CLAIM))
                    except ValueError:
                        AUTH_FAILURES.labels(
                            self.settings.internal_auth_service_name,
                            "missing_tenant_claim",
                        ).inc()
                        status_code = 403
                        await JSONResponse(
                            {"detail": "Signed tenant_id claim is required."},
                            status_code=403,
                        )(scope, receive, send)
                        return
                    if claim_tenant != header_tenant:
                        AUTH_FAILURES.labels(
                            self.settings.internal_auth_service_name,
                            "tenant_mismatch",
                        ).inc()
                        status_code = 403
                        await JSONResponse(
                            {"detail": "X-Tenant-Id does not match signed tenant_id claim."},
                            status_code=403,
                        )(scope, receive, send)
                        return
                tenant_token = _tenant_id.set(header_tenant)

            await self.app(scope, receive, capture_status)
        finally:
            if tenant_token is not None:
                _tenant_id.reset(tenant_token)
            HTTP_REQUESTS.labels(
                self.settings.internal_auth_service_name,
                method,
                normalized_path,
                str(status_code),
            ).inc()
            HTTP_DURATION.labels(
                self.settings.internal_auth_service_name,
                method,
                normalized_path,
            ).observe(time.perf_counter() - started)

    def _authenticate(self, authorization: str | None) -> dict[str, Any] | JSONResponse:
        if not self.settings.internal_auth_enabled:
            return {"sub": "auth-disabled"}
        if not self.settings.internal_auth_signing_key:
            AUTH_FAILURES.labels(self.settings.internal_auth_service_name, "server_misconfigured").inc()
            return JSONResponse({"detail": "Internal authentication is not configured."}, status_code=503)
        if not authorization or not authorization.startswith("Bearer "):
            AUTH_FAILURES.labels(self.settings.internal_auth_service_name, "missing_token").inc()
            return JSONResponse({"detail": "Missing bearer token."}, status_code=401)
        try:
            return jwt.decode(
                authorization.removeprefix("Bearer ").strip(),
                self.settings.internal_auth_signing_key,
                algorithms=["HS256"],
                audience=self.settings.internal_auth_service_name,
                issuer=self.settings.internal_auth_issuer,
                options={"require": ["exp", "iat", "iss", "aud", "sub", TENANT_CLAIM]},
            )
        except jwt.ExpiredSignatureError:
            AUTH_FAILURES.labels(self.settings.internal_auth_service_name, "expired_token").inc()
            return JSONResponse({"detail": "Expired bearer token."}, status_code=401)
        except jwt.PyJWTError:
            AUTH_FAILURES.labels(self.settings.internal_auth_service_name, "invalid_token").inc()
            return JSONResponse({"detail": "Invalid bearer token."}, status_code=401)


def tenant_index_name(settings: Any, tenant_id: str) -> str:
    canonical_tenant = normalize_tenant_id(tenant_id)
    return f"{settings.opensearch_index_prefix}-{canonical_tenant}"


def _matches(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def _normalize_path(path: str) -> str:
    path = re.sub(r"/[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", "/{id}", path)
    path = re.sub(r"/\d+", "/{id}", path)
    path = re.sub(r"/[A-Za-z0-9_-]{24,}", "/{id}", path)
    return path
