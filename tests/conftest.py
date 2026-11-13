"""Shared pytest configuration for the regression suite."""

import hashlib

# pytest uses string and bytes parameters verbatim as part of the test ID.
# Oversized-payload tests (1-2 MB bodies) therefore produced multi-megabyte
# test names, and `pytest -vv` on Windows CI prints each name straight to the
# GitHub runner's console. A single multi-megabyte line stalls the runner:
# the job never finishes, its timeouts are not enforced, and the log is lost.
# Keep long IDs short, stable and unique; the parameter values are unchanged.
MAX_PARAM_ID_CHARS = 200


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, (str, bytes)) and len(val) > MAX_PARAM_ID_CHARS:
        if isinstance(val, str):
            kind, data = "str", val.encode("utf-8", "surrogatepass")
        else:
            kind, data = "bytes", val
        digest = hashlib.sha256(data).hexdigest()[:12]
        return f"{argname}-{kind}{len(val)}-{digest}"
    return None
