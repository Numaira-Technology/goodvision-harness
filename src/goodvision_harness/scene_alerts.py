"""Compatibility alias for :mod:`.actions.alerts`."""

import sys

from .actions import alerts as _implementation

sys.modules[__name__] = _implementation
