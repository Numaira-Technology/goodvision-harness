"""Native image/text recognition client for the neural workflow block."""

from .videojev_model import VideoJevInspector


class NeuralInspector(VideoJevInspector):
    model_name = "openai/clip-vit-large-patch14"
    display_name = "Neural network recognition"
    metric_prefix = "neural_"

    def _input_frames(self, job):
        # Scene labels use the newest image; PPE assesses the full window.
        return job.frames if "ppe_violation" in job.request.event_labels else job.frames[-1:]

    def _payload(self, job):
        # Stateless requests also survive service restarts and multiple clients.
        self._template_registered = False
        return super()._payload(job)

    def _decode(self, response, expected, labels):
        if response.get("model") != self.model_name:
            raise ValueError("recognition service model mismatch")
        return super()._decode(response, expected, labels)
