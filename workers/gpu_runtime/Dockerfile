# One shared CUDA/Python layer, with versioned PyTorch layers above it.
# Contour + YOLO use 2.10; native VideoJev requires 2.8. Do not mix ABIs.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS cuda-python
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.12 python3.12-venv python3.12-dev git build-essential libgomp1 libglib2.0-0 \
    libgl1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/runtime
ENV PATH=/opt/runtime/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==25.2 setuptools==80.9.0 wheel packaging ninja

FROM cuda-python AS torch
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
      --index-url https://download.pytorch.org/whl/cu128 \
    && printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" > /opt/gpu-constraints.txt \
    && python -m pip check
# Module installs must not silently replace the shared binary ABI.
ENV PIP_CONSTRAINT=/opt/gpu-constraints.txt \
    XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor CUDA_CACHE_PATH=/tmp/cuda
