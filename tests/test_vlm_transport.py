from __future__ import annotations

import asyncio
import ssl
from pathlib import Path
from unittest.mock import Mock

import certifi
import pytest

from goodvision_harness import vlm_transport
from goodvision_harness.model_api import ModelAPIError


@pytest.fixture(autouse=True)
def clean_tls_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE"):
        monkeypatch.delenv(name, raising=False)


def test_cadata_preserves_every_trusted_certificate_and_verification() -> None:
    baseline = ssl.create_default_context(cafile=certifi.where())
    context = vlm_transport.verified_ssl_context()
    baseline_certificates = set(baseline.get_ca_certs(binary_form=True))
    assert baseline_certificates
    assert set(context.get_ca_certs(binary_form=True)) == baseline_certificates
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert not context.hostname_checks_common_name
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert context.options & ssl.OP_NO_COMPRESSION


@pytest.mark.parametrize("override", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_explicit_ca_override_preserves_file_or_directory_loading(
    monkeypatch: pytest.MonkeyPatch, override: str,
) -> None:
    context = Mock(spec=ssl.SSLContext)
    monkeypatch.setenv(override, "configured-ca-path")
    monkeypatch.setattr(vlm_transport.ssl, "SSLContext", lambda protocol: context)
    monkeypatch.setattr(
        vlm_transport.certifi, "where", Mock(side_effect=AssertionError),
    )
    assert vlm_transport.verified_ssl_context() is context
    argument = "cafile" if override == "SSL_CERT_FILE" else "capath"
    expected = {argument: "configured-ca-path"}
    context.load_verify_locations.assert_called_once_with(**expected)


def test_explicit_ca_file_takes_precedence_over_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = Mock(spec=ssl.SSLContext)
    monkeypatch.setenv("SSL_CERT_FILE", "configured-file")
    monkeypatch.setenv("SSL_CERT_DIR", "configured-directory")
    monkeypatch.setattr(vlm_transport.ssl, "SSLContext", lambda protocol: context)
    vlm_transport.verified_ssl_context()
    context.load_verify_locations.assert_called_once_with(cafile="configured-file")


def test_trust_env_false_ignores_ca_and_keylog_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "missing-directory"))
    keylog_path = tmp_path / "must-not-be-created"
    monkeypatch.setenv("SSLKEYLOGFILE", str(keylog_path))
    context = vlm_transport.verified_ssl_context(trust_env=False)
    assert context.get_ca_certs()
    assert context.keylog_filename is None
    assert not keylog_path.exists()


@pytest.mark.parametrize("missing", [False, True])
def test_invalid_ca_configuration_fails_closed_without_exposing_path_or_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: bool,
) -> None:
    path = tmp_path / "private-ca-name.pem"
    if not missing:
        path.write_text("private-invalid-certificate", encoding="ascii")
    monkeypatch.setenv("SSL_CERT_FILE", str(path))
    with pytest.raises(ModelAPIError) as caught:
        vlm_transport.verified_ssl_context()
    assert str(caught.value) == "VLM TLS trust configuration could not be loaded"
    assert caught.value.__suppress_context__


def test_client_passes_transport_settings_and_verified_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = Mock(spec=ssl.SSLContext)
    context_factory = Mock(return_value=context)
    client_factory = Mock()
    monkeypatch.setattr(vlm_transport, "verified_ssl_context", context_factory)
    monkeypatch.setattr(vlm_transport.httpx, "AsyncClient", client_factory)
    result = vlm_transport.create_vlm_http_client(
        http2=True, trust_env=False, keepalive_expiry=45.0,
    )
    assert result is client_factory.return_value
    context_factory.assert_called_once_with(trust_env=False)
    settings = client_factory.call_args.kwargs
    assert settings["verify"] is context
    assert settings["http2"] is True
    assert settings["trust_env"] is False
    assert settings["follow_redirects"] is False
    assert settings["limits"].max_connections == 3
    assert settings["limits"].max_keepalive_connections == 3
    assert settings["limits"].keepalive_expiry == 45.0


def test_default_client_has_a_caller_owned_lifetime() -> None:
    async def run() -> None:
        client = vlm_transport.create_vlm_http_client()
        assert not client.is_closed
        await client.aclose()
        assert client.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("kwargs", [
    {"http2": 1}, {"trust_env": "false"}, {"keepalive_expiry": True},
    {"keepalive_expiry": 0}, {"keepalive_expiry": -1},
    {"keepalive_expiry": float("inf")}, {"keepalive_expiry": float("nan")},
])
def test_invalid_transport_settings_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        vlm_transport.create_vlm_http_client(**kwargs)
