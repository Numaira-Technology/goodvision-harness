"""Apply identity review gates to a copied scene without changing N-frame evidence."""

from __future__ import annotations


def apply_alert_policy(scene: dict, *, stranger_enabled: bool,
                       identity_reviews: dict[str, tuple[str, float]]) -> dict:
    people = {item["subject_id"]: item for item in scene["entities"]}
    for subject, person in people.items():
        known = person.get("identity", {}).get("status") == "known"
        state = "known" if known else identity_reviews.get(subject, ("pending", 0))[0]
        person["identity_review_state"] = state
        person["identity_pending"] = (
            stranger_enabled and state not in {"known", "unknown"}
        )

    def gate(event):
        phase = event.get("phase")
        if event.get("event_label") != "stranger":
            event["alert"] = phase == "confirmed"
            return
        subjects = event.get("subject_ids", ())
        states = [people.get(sid, {}).get("identity_review_state", "pending")
                  for sid in subjects]
        if "known" in states:
            event["evidence_phase"] = phase
            event["phase"] = "retracted"
            event["alert_reason"] = "identity_known"
            event["confirmed_since_timestamp_seconds"] = None
        elif not states or any(state != "unknown" for state in states):
            event["evidence_phase"] = phase
            if phase == "confirmed":
                event["phase"] = "suspected"
            event["confirmed_since_timestamp_seconds"] = None
            event["alert_reason"] = "identity_pending"
            if event.get("review_state") == "complete":
                event["review_outcome"] = "unconfirmed"
        else:
            reviewed_at = max(identity_reviews[sid][1] for sid in subjects)
            if (phase == "confirmed"
                    and event.get("timestamp_seconds", reviewed_at) < reviewed_at):
                event["phase"] = "suspected"
            since = event.get("confirmed_since_timestamp_seconds")
            if since is not None:
                event["confirmed_since_timestamp_seconds"] = max(
                    since, reviewed_at,
                )
        event["alert"] = event.get("phase") == "confirmed"

    for event in scene["events"]:
        gate(event)
        # A compact or historical consumer must not rediscover an ungated alert.
        for observation in event.get("history", ()):
            wrapped = {**observation, "event_label": event["event_label"],
                       "subject_ids": event["subject_ids"]}
            gate(wrapped)
            observation["phase"] = wrapped["phase"]
        if (event.get("phase") == "retracted"
                and event.get("review_state") == "complete"):
            event["review_outcome"] = "retracted"
    for transition in scene.get("transitions", ()):
        gate(transition)
    return scene
