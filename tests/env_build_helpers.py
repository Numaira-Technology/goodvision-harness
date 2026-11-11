"""Small synthetic packs and measured-runtime stand-ins; never real executables."""

import io
import json
import tarfile
from types import SimpleNamespace


def runtime_facts(flash_attn=True):
    return {
        "python": "3.12",
        "torch": "2.8.0+cu128",
        "cuda": "12.8",
        "compute_capabilities": ["8.9"],
        "extensions": (
            {"flash-attn": "2.8.3.post2", "causal-conv1d": "1.7.0"}
            if flash_attn
            else {}
        ),
    }


def completed_probe(flash_attn=True):
    return SimpleNamespace(stdout=json.dumps(runtime_facts(flash_attn)))


def synthetic_pack(path, *, omit=None):
    with tarfile.open(path, "w:gz") as archive:
        for name in ("bin/python", "bin/activate", "bin/conda-unpack"):
            if name == omit:
                continue
            data = b"synthetic, non-executable review fixture\n"
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o755
            archive.addfile(member, io.BytesIO(data))
    return path
