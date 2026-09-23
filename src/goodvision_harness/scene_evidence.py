"""Compatibility import; implementation lives in :mod:`.runtime.scheduling`."""

import sys

from .runtime import scheduling as _implementation

sys.modules[__name__] = _implementation
