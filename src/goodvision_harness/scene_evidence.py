"""Compatibility alias for :mod:`.runtime.scheduling`."""

import sys

from .runtime import scheduling as _implementation

sys.modules[__name__] = _implementation
