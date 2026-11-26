"""Deterministic metadata-only wheels. No code or native binaries; never installed."""

import base64
import hashlib
import io
import zipfile


def wheel_bytes(filename, *, label=b"fixture", metadata_name=None, tag=None):
    stem, python, abi, platform = filename[:-4].rsplit("-", 3)
    name, version = stem.split("-", 1)
    info = f"{name}-{version}.dist-info/"
    files = {
        info + "METADATA": (
            f"Metadata-Version: 2.1\nName: {metadata_name or name}\n"
            f"Version: {version}\n"
        ).encode(),
        info + "WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            f"Tag: {tag or f'{python}-{abi}-{platform}'}\n"
        ).encode(),
        info + "fixture.txt": label,
    }
    rows = []
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        rows.append(f"{path},sha256={digest.decode()},{len(data)}\n")
    files[info + "RECORD"] = ("".join(rows) + info + "RECORD,,\n").encode()
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        for name, data in files.items():
            archive.writestr(zipfile.ZipInfo(name), data)
    return result.getvalue()
