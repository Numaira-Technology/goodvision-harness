"""Repair the published decord 0.6.0 Linux wheel's stale internal tag.

PyPI filename: decord-0.6.0-py3-none-manylinux2010_x86_64.whl
SHA256: 51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976
Its ctypes library is Python-independent, but WHEEL incorrectly says cp36-cp36m.
Only this known mismatch is accepted; native decoding must work before editing.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import platform
import tempfile
from pathlib import Path


def repair():
    import av
    import decord
    import numpy as np

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This wheel repair only supports Linux x86_64")
    distribution = importlib.metadata.distribution("decord")
    if distribution.version != "0.6.0":
        raise RuntimeError("Unexpected decord version; review the wheel repair")
    wheel_entry = next(
        p for p in distribution.files if str(p).endswith(".dist-info/WHEEL")
    )
    wheel = Path(distribution.locate_file(wheel_entry))
    original = wheel.read_text(encoding="utf-8")
    old = "Tag: cp36-cp36m-manylinux2010_x86_64"
    new = "Tag: py3-none-manylinux2010_x86_64"
    tags = [line for line in original.splitlines() if line.startswith("Tag:")]
    if tags not in ([old], [new]):
        raise RuntimeError(f"Unexpected decord wheel tags: {tags!r}")

    # Exercise the native library and Python binding, not just the module import.
    with tempfile.TemporaryDirectory() as temporary:
        video = Path(temporary) / "decode-smoke.mp4"
        with av.open(str(video), mode="w") as output:
            stream = output.add_stream("mpeg4", rate=5)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for brightness in (32, 128, 224):
                frame = av.VideoFrame.from_ndarray(
                    np.full((32, 32, 3), brightness, dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        frames = decord.VideoReader(str(video)).get_batch([0, 1, 2]).asnumpy()
        assert frames.shape == (3, 32, 32, 3)
        assert frames[0].mean() < frames[1].mean() < frames[2].mean()

    if tags == [old]:
        corrected = original.replace(old, new).encode("utf-8")
        wheel.write_bytes(corrected)
        record = wheel.with_name("RECORD")
        with record.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        digest = base64.urlsafe_b64encode(hashlib.sha256(corrected).digest()).rstrip(
            b"="
        )
        matching = [row for row in rows if row[0] == str(wheel_entry)]
        if len(matching) != 1:
            raise RuntimeError("Unexpected decord wheel RECORD")
        matching[0][1:] = ["sha256=" + digest.decode("ascii"), str(len(corrected))]
        with record.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
    print("decord native decoding verified; wheel metadata matches published filename")


if __name__ == "__main__":
    repair()
