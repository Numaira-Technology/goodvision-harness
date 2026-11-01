"""Private benchmark attribution survives IPC without unredacting live views."""

from dataclasses import replace

import pytest
from benchmark_runtime_fixtures import config, jpeg
from test_vlm_events_mode import DiagnosedEventInspector

from tools.benchmark_runtime.vlm_independent import IndependentVLMBranch
from tools.vlm_benchmark.provider_diagnostics import ProviderDiagnosticBus


class ProviderInspector(DiagnosedEventInspector):
    def inspect(self, job):
        verdicts = super().inspect(job)
        self.last_response_diagnostics["provider_category"] = "CoreWeave"
        return verdicts


def provider_factory(_config):
    return ProviderInspector()


def result(provider):
    return {"timing": {"response_diagnostics": {"provider_category": provider}}}


@pytest.mark.parametrize("provider", ["CoreWeave", "Reka", "Chutes", "DeepInfra"])
def test_operator_attribution_retains_only_enum_and_public_stays_redacted(provider):
    bus = ProviderDiagnosticBus(2)
    bus.publish(result(provider))
    assert (
        bus.latest().payload["timing"]["response_diagnostics"]["provider_category"]
        == "analysis engine"
    )
    rows = bus.records_after(0)
    assert (
        rows[0].payload["timing"]["response_diagnostics"]["provider_category"]
        == provider
    )
    # Mutating an operator copy cannot affect the public snapshot/history.
    rows[0].payload["timing"]["response_diagnostics"]["provider_category"] = "x"
    assert (
        bus.records_after(0)[0].payload["timing"]["response_diagnostics"][
            "provider_category"
        ]
        == provider
    )


def test_attribution_eviction_reset_and_rejected_epoch():
    bus = ProviderDiagnosticBus(2)
    for name in ("CoreWeave", "Cerebras", "Darkbloom"):
        bus.publish(result(name))
    assert list(bus._provider_labels) == [2, 3]
    assert [
        r.payload["timing"]["response_diagnostics"]["provider_category"]
        for r in bus.records_after(0)
    ] == ["Cerebras", "Darkbloom"]
    assert bus.publish(result("CoreWeave"), expected_epoch="stale") is None
    assert len(bus._provider_labels) == 2
    bus.reset()
    assert not bus._provider_labels and not bus.records_after(0)
    bus.publish(result("PRIVATE_UNKNOWN_PROVIDER"))
    assert (
        "provider_category"
        not in bus.records_after(0)[0].payload["timing"]["response_diagnostics"]
    )


def test_provider_crosses_real_worker_boundary_into_operator_record():
    branch = IndependentVLMBranch(
        replace(config(), output_mode="events"),
        inspector_factory=provider_factory,
        result_bus_factory=ProviderDiagnosticBus,
    )
    branch.start()
    try:
        for seq, at in enumerate([0, 0.4, 0.8]):
            branch.offer(jpeg(), at, seq)
        assert branch.wait_until_idle(3)
        row = branch.results.records_after(0)[0].payload
        assert row["analysis_complete"]
        assert row["timing"]["response_diagnostics"]["provider_category"] == "CoreWeave"
        assert "secret" not in str(row)
        assert (
            branch.results.latest().payload["timing"]["response_diagnostics"][
                "provider_category"
            ]
            == "analysis engine"
        )
    finally:
        branch.close(timeout_seconds=5)
