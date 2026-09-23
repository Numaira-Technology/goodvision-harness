"""Model routing and optional commodity perception adapters.

The public route API remains compatible with the former perception module.
Supervision is loaded only when its experiment is explicitly enabled.
"""

from .routes import (
    METHODS,
    ContourSessionHTTPClient,
    PerceptionSessionHTTPClient,
    ProcessingMethod,
    create_tracking_client,
    create_tracking_coordinator,
    is_contour_endpoint,
    processing_methods,
    processing_status,
)

__all__ = [
    "METHODS", "ContourSessionHTTPClient", "PerceptionSessionHTTPClient",
    "ProcessingMethod", "create_tracking_client", "create_tracking_coordinator",
    "is_contour_endpoint", "processing_methods", "processing_status",
]
