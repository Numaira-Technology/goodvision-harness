"""Identity belongs to a person; face tracks only borrow that person's state.

Raw recognition status remains in ``observation_status``. Display continuity is
not a new observation and must never contribute a stranger vote.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(slots=True)
class PersonIdentity:
    attributes: dict[str, object] = field(default_factory=dict)
    confirmed_at: float = -1.0
    embeddings: tuple[tuple[float, ...], ...] = ()

    @property
    def person_id(self) -> str | None:
        value = self.attributes.get("person_id")
        return str(value) if value else None

    def clear(self) -> None:
        self.attributes.clear()
        self.confirmed_at = -1.0
        self.embeddings = ()

    def project(self, attributes: Mapping[str, object], timestamp: float) -> dict:
        result = dict(attributes)
        if self.person_id and 0 <= self.confirmed_at <= timestamp:
            direct = (
                result.get("identity_source") == "face_match"
                and result.get("status") == "known"
                and result.get("person_id") == self.person_id
            )
            result.update(self.attributes)
            result.update(
                identity_source="face_match" if direct else "track_history",
                identity_confirmed_at=self.confirmed_at,
                identity_age_seconds=round(timestamp - self.confirmed_at, 3),
            )
        return result


def stranger_evidence(attributes: Mapping[str, object]) -> bool | None:
    """Per-person evidence: known=False, fresh unknown=True, no evidence=None.

    Defaults support direct face runtime users without a registry. A supplied
    observation/history marker always takes precedence over those defaults.
    """
    if attributes.get("face_observation_state", "observed") != "observed":
        return None
    identity = attributes.get("subject_identity")
    if attributes.get("status") == "known" or (
        isinstance(identity, Mapping) and identity.get("status") == "known"
    ):
        return False
    if (
        attributes.get("status") == "unknown"
        and attributes.get("identity_source", "face_match") == "face_match"
    ):
        return True
    return None
