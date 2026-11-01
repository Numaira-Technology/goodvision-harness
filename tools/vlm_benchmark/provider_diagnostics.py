"""Operator-only provider attribution; public live views retain redaction."""

from collections import OrderedDict

from tools.benchmark_runtime.result_bus import ResultBus
from tools.benchmark_runtime.scene_model import PUBLIC_PROVIDER_CATEGORIES


class ProviderDiagnosticBus(ResultBus):
    """Retain only the fixed provider enum alongside bounded benchmark history.

    Public latest/feature snapshots still use ResultBus privacy filtering.
    Only the operator's benchmark collector reads records_after(). No raw
    model response, endpoint, credential or arbitrary provider text is kept.
    """

    def __init__(self, history_size=128):
        super().__init__(history_size)
        self._provider_labels = OrderedDict()
        self._provider_history_limit = history_size

    def publish(self, payload, **kwargs):
        provider = (
            payload.get("timing", {})
            .get("response_diagnostics", {})
            .get("provider_category")
        )
        # ResultBus uses the same reentrant condition: a collector cannot read
        # the published record before its matching diagnostic is registered.
        with self._condition:
            record = super().publish(payload, **kwargs)
            if record is not None:
                self._provider_labels[record.cursor] = (
                    provider
                    if isinstance(provider, str)
                    and provider in PUBLIC_PROVIDER_CATEGORIES
                    else None
                )
                while len(self._provider_labels) > self._provider_history_limit:
                    self._provider_labels.popitem(last=False)
            return record

    def records_after(self, cursor, *, timeout_seconds=0.0):
        with self._condition:
            records = super().records_after(cursor, timeout_seconds=timeout_seconds)
            for record in records:
                provider = self._provider_labels.get(record.cursor)
                if provider is not None:
                    record.payload["timing"]["response_diagnostics"][
                        "provider_category"
                    ] = provider
                else:
                    record.payload.get("timing", {}).get(
                        "response_diagnostics", {}
                    ).pop("provider_category", None)
            return records

    def reset(self):
        with self._condition:
            epoch = super().reset()
            self._provider_labels.clear()
            return epoch
