"""Build only synthetic context archives locally; never contact a Docker daemon."""

import tarfile
from pathlib import Path

import pytest
from docker.utils import build
from docker.utils.fnmatch import fnmatchcase

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_FILES = (
    "demo-media/bag/synthetic.MOV",
    "demo-media/bag/synthetic.MOV.boxes.json",
    "demo-media/bag/notes.json",
)


@pytest.mark.parametrize("case_sensitive", [False, True])
@pytest.mark.parametrize(
    "ignore_file,public_files",
    [
        (
            ".dockerignore",
            (
                "src/goodvision_harness/service.py",
                "src/goodvision_harness/assets/gallery/public.png",
                "src/goodvision_harness/assets/gallery/public.jpg",
            ),
        ),
        (
            "workers/neural/Dockerfile.dockerignore",
            ("workers/neural/healthcheck.py",),
        ),
        (
            "workers/paddle_rich_label/.dockerignore",
            ("app.py", "requirements.txt"),
        ),
        ("workers/contour/.dockerignore", ("app.py", "static/public.png")),
        (
            "workers/contour/Dockerfile.engine.dockerignore",
            ("sam_scene/app.py", "contour/benchmark_gpu.py"),
        ),
        (
            "workers/contour/Dockerfile.jetson.dockerignore",
            ("sam_scene/app.py", "contour/prepare_jetson.py"),
        ),
    ],
)
def test_demo_media_stays_out_of_build_context(
    tmp_path, monkeypatch, ignore_file, public_files, case_sensitive
):
    # The SDK folds case by default. Also exercise its case-sensitive matcher
    # so Linux builds cannot hide an uppercase .MOV behind a lowercase rule.
    if case_sensitive:
        monkeypatch.setattr(build, "fnmatch", fnmatchcase)
    private_files = PRIVATE_FILES
    if ignore_file == ".dockerignore":
        # A private directory must also override the public-image exceptions.
        private_files += (
            "src/goodvision_harness/assets/demo-media/synthetic.png",
        )
    for name in (*private_files, *public_files):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic fixture only")
    patterns = [
        line.strip()
        for line in (ROOT / ignore_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    # Use Docker's context packer, not an approximation with gitignore/fnmatch.
    # The only directory supplied to it is pytest's synthetic temporary context.
    with build.tar(str(tmp_path), exclude=patterns) as archive:
        with tarfile.open(fileobj=archive) as context:
            names = set(context.getnames())
    assert not names.intersection(private_files), names.intersection(private_files)
    assert set(public_files) <= names
