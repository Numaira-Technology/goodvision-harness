"""Test real CUDA cache mechanics with untrained text weights, not accuracy."""

import json
import sys
from pathlib import Path

import sam3
import torch
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model_builder import _create_text_encoder
from sam31 import cache_person_text, verify_installed_source

verify_installed_source()
torch.manual_seed(0)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.2)
bpe = Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"
backbone = SAM3VLBackbone(torch.nn.Identity(), _create_text_encoder(str(bpe)))
# Upstream deliberately leaves these torch.empty parameters for checkpoint
# loading. Supply finite test values; otherwise this fixture can produce NaNs.
with torch.no_grad():
    torch.nn.init.normal_(
        backbone.language_backbone.encoder.positional_embedding, std=0.02,
    )
    torch.nn.init.normal_(backbone.language_backbone.encoder.text_projection, std=0.02)
backbone = backbone.cuda().eval()
parameter_bytes = sum(p.numel() * p.element_size() for p in backbone.parameters())
captions = ["person", "visual", "geometric"]
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    expected = backbone.forward_text(captions)
    repeated = backbone.forward_text(captions)
torch.cuda.synchronize()
before = torch.cuda.memory_allocated()
cache_person_text(backbone, torch)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    first = backbone.forward_text(captions)
    second = backbone.forward_text(captions)
torch.cuda.synchronize()
report = {
    "scope": "untrained_text_encoder_cuda_cache_mechanics",
    "checkpoint_loaded": False,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "encoder_parameter_bytes": parameter_bytes,
    "cuda_allocated_before": before,
    "cuda_allocated_after": torch.cuda.memory_allocated(),
    "results": {
        key: {"shape": list(value.shape), "dtype": str(value.dtype),
              "finite": bool(torch.isfinite(value).all()),
              "uncached_repeat_equal": torch.equal(expected[key], repeated[key]),
              "equal": torch.equal(expected[key], value),
              "independent_storage": value.data_ptr() != second[key].data_ptr()}
        for key, value in first.items()
    },
    "all_text_parameters_meta": all(p.is_meta for p in backbone.parameters()),
}
report["ok"] = report["all_text_parameters_meta"] and all(
    row["finite"] and row["equal"] and row["independent_storage"]
    for row in report["results"].values()
)
with Path(sys.argv[1]).open("x") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")
print(json.dumps(report), flush=True)
raise SystemExit(0 if report["ok"] else 1)
