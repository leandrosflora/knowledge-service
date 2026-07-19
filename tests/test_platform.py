from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.platform import normalize_tenant_id, tenant_index_name


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
