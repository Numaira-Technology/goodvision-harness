"""Compatibility import for the Jetson build-time specialization."""

from __future__ import annotations

if __package__:
    from ..sam_scene.sam31 import cache_person_text as cache_person_text
else:
    from sam31 import cache_person_text as cache_person_text
