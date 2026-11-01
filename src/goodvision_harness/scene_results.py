"""Compatibility alias for :mod:`.actions.results`."""

import sys

from .actions import results as _implementation

sys.modules[__name__] = _implementation
