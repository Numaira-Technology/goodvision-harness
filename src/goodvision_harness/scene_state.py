"""Compatibility import; implementation lives in :mod:`.events.state_machine`."""

import sys

from .events import state_machine as _implementation

sys.modules[__name__] = _implementation
