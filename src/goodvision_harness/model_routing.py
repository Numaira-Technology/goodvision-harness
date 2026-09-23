"""Compatibility import; implementation lives in :mod:`.perception.routing`."""

import sys

from .perception import routing as _implementation

sys.modules[__name__] = _implementation
