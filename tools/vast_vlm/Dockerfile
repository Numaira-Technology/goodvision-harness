# Optional custom Vast template. Build/push only when separately requested.
# Base is v0.31.0, resolved 2026-10-17; never silently upgrade to latest/nightly.
FROM vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
WORKDIR /workspace/harness
# Build context is the exported source bundle, not the developer's full checkout.
COPY . /workspace/harness
# Vast's SSH entrypoint supplies access. Model launch is explicit after preparation.
ENTRYPOINT []
CMD ["sleep", "infinity"]
