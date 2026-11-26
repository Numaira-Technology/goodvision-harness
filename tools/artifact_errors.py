"""Safe diagnostic labels for immutable artifact operations."""

from collections.abc import Mapping


def upload_failure_category(error):
    """Return a fixed safe label; never format arbitrary SDK fields or types."""
    categories = {
        "AccessDenied": "AccessDenied",
        "InvalidAccessKeyId": "InvalidAccessKeyId",
        "SignatureDoesNotMatch": "SignatureDoesNotMatch",
        "PreconditionFailed": "PreconditionFailed",
        "ConditionalRequestConflict": "ConditionalRequestConflict",
        "RequestTimeout": "RequestTimeout",
        "SlowDown": "SlowDown",
        "InternalError": "InternalError",
        "ServiceUnavailable": "ServiceUnavailable",
    }
    try:
        response = getattr(error, "response", None)
        detail = response.get("Error") if isinstance(response, Mapping) else None
        code = detail.get("Code") if isinstance(detail, Mapping) else None
        if type(code) is str:
            return categories.get(code, "unclassified error")
    except Exception:
        # Malformed diagnostics must not mask an uncertain write outcome.
        pass
    return "unclassified error"
