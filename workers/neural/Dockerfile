# First standalone CLIP release: preserve the V2.0 application and CUDA ABI.
# This is a dependency layer, not a request to start the VideoJev container.
# Verify the base image identity with package_neural_container.py before delivery.
ARG NEURAL_BASE_IMAGE=goodvision-system1:pr75-603f20d
FROM ${NEURAL_BASE_IMAGE}

LABEL org.opencontainers.image.title="GoodVision Neural (CLIP)" \
    org.opencontainers.image.version="v2.0" \
    org.opencontainers.image.revision="603f20dc0089cc0b0f21429d8432f3c32aa4b2a6" \
    org.opencontainers.image.source="https://github.com/Numaira-Technology/goodvision-harness" \
    com.goodvision.module="goodvision-neural" \
    com.goodvision.base.image="goodvision-system1:pr75-603f20d" \
    com.goodvision.base.id="sha256:b7e2098b9aa4e6dcf98810797d490df5993027db31e476ad22d326e73750f779"

COPY workers/neural/healthcheck.py /opt/goodvision-neural/healthcheck.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/huggingface \
    HF_HUB_CACHE=/models/huggingface PYTHONUNBUFFERED=1
EXPOSE 8098
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=6 \
    CMD ["python", "/opt/goodvision-neural/healthcheck.py"]
ENTRYPOINT ["python", "-m", "goodvision_harness.neural_service"]
CMD ["--host", "0.0.0.0", "--port", "8098", "--cache-dir", "/models/huggingface"]
