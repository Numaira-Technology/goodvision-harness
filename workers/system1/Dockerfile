ARG GPU_RUNTIME_IMAGE=goodvision-gpu-runtime:torch2.8-cu128
FROM ${GPU_RUNTIME_IMAGE}
WORKDIR /app
COPY tools/videojev_probe/requirements.txt /tmp/module-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/module-requirements.txt \
    && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0+PTX' \
       python -m pip install flash-attn==2.8.3.post2 causal-conv1d==1.7.0 --no-build-isolation
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install . && python -m pip check
ENV HF_HUB_OFFLINE=1 HF_HOME=/data/huggingface
EXPOSE 8097
CMD ["python", "-m", "goodvision_harness.videojev_service", "--host", "0.0.0.0", "--port", "8097", "--cache-dir", "/models/huggingface", "--result-log", "/data/results.jsonl"]
