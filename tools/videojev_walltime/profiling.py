"""Capture operator activity for one explicit real single-window inference."""

import json
from pathlib import Path
import sys

from tools.videojev_native.runtime import VideoJevRuntime


def _profile_error(stage, error):
    print(json.dumps({"profile_error": str(error), "profile_stage": stage}),
          file=sys.stderr, flush=True)


class ProfiledVideoJevRuntime(VideoJevRuntime):
    """Keep profiling outside the default runtime and ordinary model inputs."""

    def __init__(self, *args, profile_request, profile_output, **kwargs):
        super().__init__(*args, **kwargs)
        self._profile_request = profile_request
        self._profile_output = Path(profile_output)
        self._profile_calls = 0

    def predict_prepared(self, *args, **kwargs):
        self._profile_calls += 1
        if self._profile_calls != self._profile_request:
            return super().predict_prepared(*args, **kwargs)

        try:
            from torch.profiler import ProfilerActivity, profile

            trace = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            )
            trace.__enter__()
        except Exception as error:
            _profile_error("start", error)
            return super().predict_prepared(*args, **kwargs)

        try:
            result = super().predict_prepared(*args, **kwargs)
        finally:
            # Keep a model exception intact. A profiler stop error must not
            # turn an already successful inference into a failed window.
            try:
                trace.__exit__(*sys.exc_info())
            except Exception as error:
                _profile_error("stop", error)

        # Export after inference. An export error must not discard its result
        # or resend the window; the service keeps its existing failure handling.
        try:
            self._profile_output.mkdir(parents=True, exist_ok=True)
            trace.export_chrome_trace(str(self._profile_output / "trace.json"))
            operators = trace.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=80,
            )
            (self._profile_output / "operators.txt").write_text(operators, encoding="utf-8")
        except Exception as error:
            _profile_error("export", error)
        return result
