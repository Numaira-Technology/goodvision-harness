"""Verified, reusable HTTP transport for latency-sensitive VLM requests."""

from __future__ import annotations

import math
import os
import ssl
from pathlib import Path

import certifi
import httpx

from .model_api import ModelAPIError


def verified_ssl_context(trust_env: bool = True) -> ssl.SSLContext:
    """Load the normal certifi trust anchors without OpenSSL's slow Windows file I/O.

    Explicit environment CA overrides retain their file/directory loading path.
    A malformed override never falls back to a different trust store. OpenSSL
    loads hashed-directory anchors at handshake time as usual. Contexts are
    independent so trust configuration changes take effect on the next client.
    """

    if not isinstance(trust_env, bool):
        raise ValueError("trust_env must be boolean")
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.hostname_checks_common_name = False
        if hasattr(context, "post_handshake_auth"):
            context.post_handshake_auth = True
        ca_file = os.environ.get("SSL_CERT_FILE") if trust_env else None
        ca_directory = os.environ.get("SSL_CERT_DIR") if trust_env else None
        if ca_file:
            context.load_verify_locations(cafile=ca_file)
        elif ca_directory:
            context.load_verify_locations(capath=ca_directory)
        else:
            pem = Path(certifi.where()).read_text(encoding="ascii")
            context.load_verify_locations(cadata=pem)
        keylog_file = os.environ.get("SSLKEYLOGFILE") if trust_env else None
        if keylog_file:
            context.keylog_filename = keylog_file
        return context
    except (OSError, UnicodeError, ValueError):
        # SSL and file exceptions can expose user CA paths or certificate data.
        raise ModelAPIError("VLM TLS trust configuration could not be loaded") from None


def create_vlm_http_client(
    *,
    http2: bool = False,
    trust_env: bool = True,
    keepalive_expiry: float = 60.0,
) -> httpx.AsyncClient:
    """Create a verified three-connection pool; the caller owns its lifetime."""

    if not isinstance(http2, bool):
        raise ValueError("http2 must be boolean")
    if not isinstance(trust_env, bool):
        raise ValueError("trust_env must be boolean")
    if (
        isinstance(keepalive_expiry, bool)
        or not isinstance(keepalive_expiry, (int, float))
        or not math.isfinite(keepalive_expiry)
        or keepalive_expiry <= 0
    ):
        raise ValueError("keepalive_expiry must be finite and positive")
    return httpx.AsyncClient(
        verify=verified_ssl_context(trust_env=trust_env),
        http2=http2,
        trust_env=trust_env,
        limits=httpx.Limits(
            max_connections=3, max_keepalive_connections=3,
            keepalive_expiry=keepalive_expiry,
        ),
        follow_redirects=False,
    )
