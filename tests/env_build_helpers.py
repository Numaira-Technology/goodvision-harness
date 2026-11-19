"""Small synthetic packs and measured-runtime stand-ins; never real executables."""

import base64
import hashlib
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
        site = "lib/python3.12/site-packages/"
        metadata = b"Name: synthetic-review-fixture\nVersion: 1\n"
        record = (
            "synthetic_review_fixture-1.dist-info/METADATA,sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(metadata).digest())
            .rstrip(b"=")
            .decode()
            + f",{len(metadata)}\n"
        ).encode()
        for name, data in (("METADATA", metadata), ("RECORD", record)):
            member = tarfile.TarInfo(
                site + "synthetic_review_fixture-1.dist-info/" + name
            )
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def corrupt_package_metadata(path):
    """Keep a valid archive layout while making its RECORD digest disagree."""
    with tarfile.open(path, "r:gz") as archive:
        entries = [(member, archive.extractfile(member).read()) for member in archive]
    with tarfile.open(path, "w:gz") as archive:
        for member, data in entries:
            if member.name.endswith("/METADATA"):
                data = data.replace(b"Name:", b"Namy:")
            archive.addfile(member, io.BytesIO(data))
