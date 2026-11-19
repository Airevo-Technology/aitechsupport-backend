"""Marketplace name shown to customers follows AIREVO_API_BASE (airevo.my vs airevo.com.au)."""
import pytest

from app.core import airevo


@pytest.mark.parametrize("base,want", [
    ("", "airevo.com.au"),  # default API base
    ("https://airevo.my/api/marketplace/v1", "airevo.my"),
    ("https://www.airevo.my/api/marketplace/v1/", "airevo.my"),
    ("https://staging.airevo.com.au/api/marketplace/v1", "staging.airevo.com.au"),
])
def test_site_name_follows_api_base(monkeypatch, base, want):
    monkeypatch.setattr(airevo.config_store, "get", lambda k, d="": base if k == "AIREVO_API_BASE" else d)
    assert airevo.site_name() == want
