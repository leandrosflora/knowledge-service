from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from starlette.responses import JSONResponse

from app.platform import PlatformMiddleware, normalize_tenant_id, tenant_index_name

CALLER = "agent-runtime-renegotiation"
AUDIENCE = "knowledge-service"
ISSUER = "conversational-ai-platform"
INBOUND_SECRET = "a" * 32
TENANT_ID = "00000000-0000-0000-0000-000000000001"


def _settings(**overrides) -> SimpleNamespace:
    defaults = dict(
        internal_auth_enabled=True,
        internal_auth_service_name=AUDIENCE,
        internal_auth_issuer=ISSUER,
        internal_auth_inbound_secrets={CALLER: INBOUND_SECRET},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _token(*, kid: str, sub: str, secret: str, aud: str = AUDIENCE, iss: str = ISSUER) -> str:
    now = int(time.time())
    claims = {
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "tenant_id": TENANT_ID,
        "iat": now,
        "exp": now + 300,
    }
    return jwt.encode(claims, secret, algorithm="HS256", headers={"kid": kid})


def _middleware(settings: SimpleNamespace) -> PlatformMiddleware:
    return PlatformMiddleware(lambda *a, **kw: None, settings=settings)


def test_normalize_tenant_rejects_non_uuid() -> None:
    with pytest.raises(ValueError, match="UUID"):
        normalize_tenant_id("tenant-a")


def test_normalize_tenant_rejects_empty_uuid() -> None:
    with pytest.raises(ValueError, match="empty UUID"):
        normalize_tenant_id("00000000-0000-0000-0000-000000000000")


def test_index_name_uses_lossless_canonical_uuid() -> None:
    settings = SimpleNamespace(opensearch_index_prefix="faq_chunks")

    index_name = tenant_index_name(
        settings,
        "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
    )

    assert index_name == "faq_chunks-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_distinct_tenants_never_share_index_name() -> None:
    settings = SimpleNamespace(opensearch_index_prefix="faq_chunks")

    first = tenant_index_name(settings, "00000000-0000-0000-0000-000000000001")
    second = tenant_index_name(settings, "00000000-0000-0000-0000-000000000002")

    assert first != second


def test_authenticate_accepts_token_from_allow_listed_caller() -> None:
    middleware = _middleware(_settings())
    token = _token(kid=CALLER, sub=CALLER, secret=INBOUND_SECRET)

    result = middleware._authenticate(f"Bearer {token}")

    assert isinstance(result, dict)
    assert result["sub"] == CALLER


def test_authenticate_rejects_kid_outside_allow_list() -> None:
    middleware = _middleware(_settings())
    token = _token(kid="whatsapp-bff", sub="whatsapp-bff", secret="b" * 32)

    result = middleware._authenticate(f"Bearer {token}")

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


def test_authenticate_rejects_allow_listed_kid_with_wrong_signature() -> None:
    middleware = _middleware(_settings())
    token = _token(kid=CALLER, sub=CALLER, secret="wrong-secret-that-is-32-bytes!!")

    result = middleware._authenticate(f"Bearer {token}")

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


def test_authenticate_rejects_kid_sub_mismatch() -> None:
    middleware = _middleware(_settings())
    token = _token(kid=CALLER, sub="tool-service-renegotiation", secret=INBOUND_SECRET)

    result = middleware._authenticate(f"Bearer {token}")

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


def test_authenticate_rejects_missing_kid_header() -> None:
    middleware = _middleware(_settings())
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": CALLER,
            "tenant_id": TENANT_ID,
            "iat": now,
            "exp": now + 300,
        },
        INBOUND_SECRET,
        algorithm="HS256",
    )

    result = middleware._authenticate(f"Bearer {token}")

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


def test_authenticate_rejects_when_no_inbound_secrets_configured() -> None:
    middleware = _middleware(_settings(internal_auth_inbound_secrets={}))
    token = _token(kid=CALLER, sub=CALLER, secret=INBOUND_SECRET)

    result = middleware._authenticate(f"Bearer {token}")

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


def test_authenticate_bypassed_when_auth_disabled() -> None:
    middleware = _middleware(_settings(internal_auth_enabled=False))

    result = middleware._authenticate(None)

    assert result == {"sub": "auth-disabled"}
