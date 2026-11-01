"""Native image/text recognition client for the neural workflow block."""

from .videojev_model import VideoJevInspector


class NeuralInspector(VideoJevInspector):
    model_name = "openai/clip-vit-large-patch14"
    display_name = "Neural network recognition"
    metric_prefix = "neural_"

    def _payload(self, job):
        # Stateless requests also survive service restarts and multiple clients.
        self._template_registered = False
        return super()._payload(job)

    def _decode(self, response, expected, labels):
        if response.get("model") != self.model_name:
            raise ValueError("recognition service model mismatch")
        return super()._decode(response, expected, labels)
