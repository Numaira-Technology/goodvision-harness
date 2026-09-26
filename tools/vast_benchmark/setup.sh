#!/usr/bin/env bash
# Configure an already rented Linux instance. Never creates or pays for instances.
set -euo pipefail
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH_ROOT"
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo 'This setup is for the rented Linux x86_64 instance, not the Mac.' >&2
  exit 2
fi
PYTHON_BIN="${BENCH_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
COMPONENT="${1:-}"
case "$COMPONENT" in
  client)
    "$PYTHON_BIN" -m venv .venv-harness-benchmark
    .venv-harness-benchmark/bin/python -m pip install -e '.[video]'
    .venv-harness-benchmark/bin/python tools/sam31_latency_run.py --help
    exit 0
    ;;
  seg) ENV_DIR=.venv-seg-benchmark ;;
  sam) ENV_DIR=.venv-sam31-benchmark ;;
  *) echo 'Usage: bash tools/vast_benchmark/setup.sh seg|sam|client' >&2; exit 2 ;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" -m venv "$ENV_DIR"
PYTHON_EXE="$BENCH_ROOT/$ENV_DIR/bin/python"
"$PYTHON_EXE" -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126
if [ "$COMPONENT" = sam ]; then
  "$PYTHON_EXE" -m pip install -r workers/sam_scene/requirements.txt
  "$PYTHON_EXE" -m pip install \
    'git+https://github.com/facebookresearch/sam3.git@660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7'
else
  "$PYTHON_EXE" -m pip install -r tools/vast_benchmark/requirements-seg.txt
  "$PYTHON_EXE" -m pip install fastapi==0.141.1 uvicorn==0.34.2 Pillow==11.2.1 av==17.1.0
fi
"$PYTHON_EXE" -m pip check
"$PYTHON_EXE" - <<'PY'
import torch
assert torch.__version__ == '2.10.0+cu126', torch.__version__
assert torch.cuda.is_available(), 'CUDA unavailable: verify host driver and GPU allocation'
print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)
PY
if [ "$COMPONENT" = seg ]; then
  "$PYTHON_EXE" -m tools.seg_benchmark doctor
  "$PYTHON_EXE" tools/vast_benchmark/install_services.py
  echo 'Runtime installed. Candidate supervisor services are registered but stopped.'
else
  "$PYTHON_EXE" -c 'from sam3.model_builder import build_sam3_multiplex_video_predictor; print("SAM3.1 import OK; weights not loaded")'
  echo 'Runtime installed. No benchmark or model service was started.'
fi
