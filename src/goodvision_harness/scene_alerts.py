"""Compatibility import; implementation lives in :mod:`.actions.alerts`."""

import sys

from .actions import alerts as _implementation

sys.modules[__name__] = _implementation
