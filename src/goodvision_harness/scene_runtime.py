"""Compatibility alias for :mod:`.runtime.scene`."""

import sys

from .runtime import scene as _implementation

sys.modules[__name__] = _implementation
