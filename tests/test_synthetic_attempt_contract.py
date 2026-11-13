"""Invalid synthetic attempts must fail without replacing their evidence."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name,function_name,extra",
    [
        (
            "test_tracked_grounding",
            "test_twelve_sources_through_http_broker_and_spawned_workers",
            {"load_profile": "historical"},
        ),
        (
            "test_tracked_decisions_flow",
            "test_twelve_sources_native_decisions_and_person_reports",
            {},
        ),
        (
            "test_vast_fp8_flow",
            "test_twelve_sources_fp8_and_person_reports",
            {"sequences": None},
        ),
    ],
)
def test_invalid_first_stage_is_not_replaced(
    tmp_path, monkeypatch, request, module_name, function_name, extra,
):
    module = importlib.import_module(module_name)
    attempts = []

    class Broker:
        endpoint = "http://synthetic.invalid"

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, *args):
            raise AssertionError("no transport should run in this contract test")

        def drain(self):
            return {"submitted": 0, "ready": 0}

    def invalid_stage(*args, **kwargs):
        attempts.append(args[4])
        if "source_diagnostics" in kwargs:
            kwargs["source_diagnostics"]["attempt"] = len(attempts)
        return {
            "input_timing_valid": False,
            "input_release_lag_seconds": {"max": 1},
            "sender_missed_samples": 1,
            "sender_missed_samples_by_stream": [1],
            "offer_rejected_by_stream": [0],
            "all_drained": False,
            "result_records_missing": 1,
        }

    monkeypatch.setattr(module, "DetectorBroker", Broker)
    monkeypatch.setattr(module, "run_stage", invalid_stage)
    monkeypatch.setattr(module, "jpeg", lambda: b"synthetic frame")
    arguments = dict(tmp_path=tmp_path, monkeypatch=monkeypatch, mode="events", **extra)
    if module_name != "test_vast_fp8_flow":
        arguments["request"] = request
    if module_name == "test_tracked_grounding":
        arguments["fake_models"] = ("http://synthetic.invalid", {"detector": []})
    with pytest.raises(AssertionError):
        getattr(module, function_name)(**arguments)
    assert attempts == [tmp_path]
    assert not (tmp_path / "retry").exists()
