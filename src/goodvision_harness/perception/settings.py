"""Strict, dependency-free configuration for the optional Supervision experiment."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..observations.schema import seconds, text


@dataclass(frozen=True, slots=True)
class Zone:
    name: str
    points: tuple[tuple[float, float], ...]

    def validate(self, *, line: bool) -> None:
        text(self.name, "zone name")
        if not isinstance(self.points, tuple) or not (
            len(self.points) == 2 if line else 3 <= len(self.points) <= 128
        ):
            raise ValueError("line needs two points; polygon needs 3..128")
        for point in self.points:
            if not isinstance(point, tuple) or len(point) != 2:
                raise ValueError("zone points must be normalized xy pairs")
            for value in point:
                seconds(value, "zone coordinate")
                if value > 1:
                    raise ValueError("zone coordinates must be within 0..1")
        if len(set(self.points)) != len(self.points):
            raise ValueError("zone points must be distinct")
        if (
            not line
            and abs(
                sum(
                    self.points[i - 1][0] * point[1] - point[0] * self.points[i - 1][1]
                    for i, point in enumerate(self.points)
                )
            )
            < 1e-9
        ):
            raise ValueError("polygon must enclose area")


@dataclass(frozen=True, slots=True)
class SupervisionSettings:
    enabled: bool = False
    confidence_threshold: float | None = None
    nms_threshold: float | None = None
    smoothing_length: int = 5
    max_gap_seconds: float = 3.0
    max_tracks: int = 2048
    zones: tuple[Zone, ...] = ()
    lines: tuple[Zone, ...] = ()

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("supervision.enabled must be boolean")
        for name in ("confidence_threshold", "nms_threshold"):
            value = getattr(self, name)
            if value is not None:
                seconds(value, name)
                if value > 1:
                    raise ValueError(f"{name} must be within 0..1")
        for name, maximum in (("smoothing_length", 32), ("max_tracks", 65536)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be within 1..{maximum}")
        seconds(self.max_gap_seconds, "max_gap_seconds")
        if not 0 < self.max_gap_seconds <= 90:
            raise ValueError("max_gap_seconds must be within 0..90")
        for zones, line in ((self.zones, False), (self.lines, True)):
            if not isinstance(zones, tuple) or len(zones) > 16:
                raise ValueError("at most 16 polygons and 16 lines are supported")
            if any(not isinstance(zone, Zone) for zone in zones):
                raise ValueError("zones must contain Zone entries")
            if len({z.name for z in zones}) != len(zones):
                raise ValueError("zone names must be unique")
            for zone in zones:
                zone.validate(line=line)

    @classmethod
    def from_mapping(cls, value: Mapping) -> "SupervisionSettings":
        if not isinstance(value, Mapping):
            raise ValueError("supervision must be a mapping")
        if set(value) - cls.__dataclass_fields__.keys():
            raise ValueError("unknown supervision setting")
        kwargs = dict(value)
        for key in ("zones", "lines"):
            if key not in kwargs:
                continue
            entries = kwargs[key]
            if not isinstance(entries, list):
                raise ValueError(f"{key} must be a list")
            parsed = []
            for entry in entries:
                if not isinstance(entry, Mapping) or set(entry) != {"name", "points"}:
                    raise ValueError("zones require name and points")
                points = entry["points"]
                if not isinstance(points, list) or any(
                    not isinstance(p, list) for p in points
                ):
                    raise ValueError("zone points must be a list of xy pairs")
                parsed.append(Zone(entry["name"], tuple(tuple(p) for p in points)))
            kwargs[key] = tuple(parsed)
        return cls(**kwargs)
