"""One-time gate/up projection merging for the pinned bias-free language MLP."""

from torch import cat, nn


class FusedGateUpMLP(nn.Module):
    def __init__(self, original):
        super().__init__()
        gate, up = original.gate_proj, original.up_proj
        # Allocate only the concatenated real weights, without random CUDA init.
        self.gate_up_proj = nn.Linear(
            gate.in_features, gate.out_features + up.out_features,
            bias=False, device="meta", dtype=gate.weight.dtype,
        )
        self.gate_up_proj.weight = nn.Parameter(
            cat((gate.weight, up.weight), dim=0), requires_grad=False,
        )
        self.act_fn = original.act_fn
        self.down_proj = original.down_proj
        self.train(original.training)

    def forward(self, hidden_states):
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)
