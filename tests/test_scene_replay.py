"""Offline replay proves event continuity and rejects malformed replay inputs."""

import io
import json
from pathlib import Path

import pytest

from tools.scene_replay import MAX_LINE_BYTES, ReplayError, main, replay

FIXTURE = Path(__file__).parent / "fixtures" / "scene_smoking.jsonl"
HEADER = {"type": "session", "stream_session_id": "test", "source_id": "offline"}


def _source(*rows):
    return io.BytesIO("\n".join(json.dumps(row) for row in rows).encode("utf-8"))


def _event(snapshot):
    return snapshot["scene"]["events"][0]


def test_smoking_replay_tracks_one_episode_through_occlusion_and_clear():
    with FIXTURE.open("rb") as source:
        rows = list(replay(source))
    assert len(rows) == 17
    assert _event(rows[3])["phase"] == "suspected"
    assert _event(rows[5])["phase"] == "confirmed"
    assert _event(rows[6])["phase"] == "confirmed"
    assert _event(rows[6])["observation_state"] == "uncertain"
    assert rows[8]["accepted"] is False  # A predicted box is no observation.
    assert rows[9]["accepted"] is False  # An unseen negative cannot clear Alice.
    assert _event(rows[9])["phase"] == "confirmed"
    assert _event(rows[11])["phase"] == "confirmed"
    assert _event(rows[13])["phase"] == "ended"
    assert rows[14]["accepted"] is False  # Delayed positive cannot resurrect it.
    assert rows[15]["accepted"] is False  # Duplicate evidence cannot confirm it.
    assert _event(rows[-1])["phase"] == "ended"
    assert len({_event(row)["event_id"] for row in rows[3:]}) == 1
    assert all(
        event["subject_ids"] == ["alice"]
        for row in rows
        for event in row["scene"]["events"]
    )
    people = {person["subject_id"]: person for person in rows[-1]["scene"]["entities"]}
    assert set(people) == {"alice", "bob"}
    assert people["alice"]["identity"]["person_id"] == "employee-alice"


@pytest.mark.parametrize("source", [b"", b"{}", b"[]", b"null", b"\n"])
def test_replay_requires_session_header(source):
    with pytest.raises(ReplayError, match="line 1"):
        list(replay(io.BytesIO(source)))


@pytest.mark.parametrize(
    "record",
    [
        HEADER,
        {"type": "tick", "timestamp_seconds": True},
        {"type": "tick", "timestamp_seconds": -1},
        {"type": "tick", "timestamp_seconds": float("nan")},
        {"type": "tick", "timestamp_seconds": 1, "surprise": 2},
        {"type": "event", "timestamp_seconds": 1},
        {
            "type": "person",
            "subject_id": "a",
            "timestamp_seconds": 1,
            "frame_sequence": 1,
            "bbox": [0, 0, 2, 1],
        },
        {
            "type": "person",
            "subject_id": "a",
            "timestamp_seconds": 1,
            "frame_sequence": True,
            "bbox": [0, 0, 1, 1],
        },
    ],
)
def test_invalid_record_is_a_line_numbered_error(record):
    with pytest.raises(ReplayError, match="line 2"):
        list(replay(_source(HEADER, record)))


def test_reader_rejects_oversized_records_without_unbounded_read():
    class BoundedReader(io.BytesIO):
        def readline(self, size=-1):
            assert size == MAX_LINE_BYTES + 1
            return super().readline(size)

    with pytest.raises(ReplayError, match="exceeds 1 MiB"):
        list(replay(BoundedReader(b" " * (MAX_LINE_BYTES + 1))))


def test_duplicate_keys_and_invalid_utf8_are_rejected():
    for raw in (b'{"type":"session","type":"tick"}', b"\xff"):
        with pytest.raises(ReplayError, match="line 1: invalid UTF-8 JSON"):
            list(replay(io.BytesIO(raw)))


def test_cli_writes_snapshots_and_preserves_existing_output(tmp_path, capsys):
    output = tmp_path / "scene.jsonl"
    assert main([str(FIXTURE), "--output", str(output)]) == 0
    original = output.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    assert rows[-1]["scene"]["events"][0]["phase"] == "ended"
    assert main([str(FIXTURE), "--output", str(output)]) == 2
    assert output.read_bytes() == original
    assert "output already exists" in capsys.readouterr().err


def test_cli_invalid_input_returns_nonzero_and_concise_message(tmp_path, capsys):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"private_payload":"do not echo this"}', encoding="utf-8")
    assert main([str(source)]) == 2
    captured = capsys.readouterr()
    assert "line 1" in captured.err
    assert "private_payload" not in captured.err
    assert captured.out == ""


def test_cli_missing_input_does_not_create_output(tmp_path, capsys):
    output = tmp_path / "output.jsonl"
    assert main([str(tmp_path / "missing"), "--output", str(output)]) == 2
    assert not output.exists()
    assert "could not read input or write output" in capsys.readouterr().err
