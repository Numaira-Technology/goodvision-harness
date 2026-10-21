"""Read-only view of backend tracks; no browser-driven detector or model calls."""

from tools.vlm_benchmark.monitor import LiveMonitor, lane_name


class PersonMonitor(LiveMonitor):
    broker = None

    def snapshot(self):
        data = super().snapshot()
        if data.get("stage") and self.broker is not None:
            with self.broker.condition:
                for i, camera in enumerate(data["cameras"]):
                    row = self.broker.latest.get(i)
                    if row and row["state"] == "ready":
                        camera["people"] = {
                            "sequence": row["frame_sequence"],
                            "timestamp": row["timestamp_seconds"],
                            "objects": row["people"],
                        }
        return data

    def result(self, camera, report):
        super().result(camera, report)
        if report.get("subject_id_scope") != "stream_session":
            return
        with self.lock:
            view = self.state["cameras"][camera]["lanes"].get(lane_name(report))
            if not view or view["id"] != report["message_id"]:
                return  # Historical results stay in logs, never roll the UI back.
            ids = {
                o["object_id"]: o["attributes"]["track_id"]
                for o in report["objects"]
                if o.get("attributes", {}).get("track_id")
            }
            view["person_judgments"] = [
                {
                    "id": sid,
                    "events": {
                        o["event_label"]: o["detected"]
                        for o in report["observations"]
                        if oid in o["subject_ids"]
                    },
                }
                for oid, sid in ids.items()
            ]
