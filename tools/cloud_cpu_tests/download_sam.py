"""Download the same SAM checkpoint as main's rental script, on an existing host."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "facebook/sam3.1"
FILENAME = "sam3.1_multiplex.pt"
MIN_CHECKPOINT_BYTES = 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / ".models/sam3.1")
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    checkpoint = directory / FILENAME
    if checkpoint.is_file() and checkpoint.stat().st_size >= MIN_CHECKPOINT_BYTES:
        print(f"Using existing checkpoint: {checkpoint}")
        return
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        parser.error("Set HF_TOKEN for an account able to access facebook/sam3.1")
    # The rental script also enables network access for this explicit download.
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=REPOSITORY, filename=FILENAME, local_dir=directory, token=token
        )
    except Exception as error:
        raise SystemExit(
            f"SAM download failed ({type(error).__name__}). "
            "Check model access, HF_TOKEN, connectivity and free disk space."
        ) from None
    if Path(path).stat().st_size < MIN_CHECKPOINT_BYTES:
        raise SystemExit("Downloaded checkpoint is too small to be a SAM model")
    print(f"Checkpoint ready: {path}")


if __name__ == "__main__":
    main()
