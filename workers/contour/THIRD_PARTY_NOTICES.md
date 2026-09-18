# Third-party components

GoodVision Contour is the service, API and console branding. It does not claim
authorship or ownership of the underlying third-party model.

The optional GPU runtime integrates **Meta SAM 3.1** from
[`facebookresearch/sam3`](https://github.com/facebookresearch/sam3), pinned at
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`, and the separately obtained
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) checkpoint.
These materials are governed by the
[SAM License](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE).

The runtime Docker build preserves the original model LICENSE verbatim at
`/opt/notices/model/LICENSE`. The authorized download helper preserves the model
repository's LICENSE beside the checkpoint. When distributing weights or the
runtime, include those licenses, origin metadata, and any required notices.
Do not remove them to change product branding.

The lightweight console image contains no model weights or model implementation.
Its web stack includes FastAPI, Starlette, Pydantic, HTTPX and Uvicorn, whose
licenses remain in the installed distributions. The GPU image also retains the
notices bundled with its runtime dependencies. Direct dependencies are pinned;
the runtime is not yet a complete transitive, hash-locked release.

The runtime includes decord 0.6.0. Its published Linux x86_64 wheel is named
`decord-0.6.0-py3-none-manylinux2010_x86_64.whl`, but its internal WHEEL metadata
incorrectly declares `cp36-cp36m`. `repair_decord_metadata.py` corrects only this
known tag mismatch and updates RECORD; it leaves the native library unchanged.
The build requires a successful native video decode before applying the repair,
then runs the normal dependency check. The original dependency license is retained.

The Jetson variant uses NVIDIA's JetPack-compatible iGPU framework container and
does not install decord; this service submits decoded frames, not file paths.
`prepare_jetson.py` applies loading and memory-management changes to the pinned
model builder and internal workers: meta-device construction, memory-mapped
checkpoint assignment, elimination of a discarded tracker backbone, lazy
positional caches, a CUDA allocation budget, and release of unused allocator
workspace. Small configuration scalars are generated on the CPU and the
nonpersistent text attention mask is reconstructed using upstream code.
Runtime tensor weights use BF16; decoder feed-forward layers that explicitly
disable autocast stay FP32 and their inputs are promoted to FP32. Vector
parameters and buffers retain their original types, including complex RoPE
buffers. The original checkpoint on disk is unchanged, strict validation remains,
and autocast's duplicate-weight cache is disabled. This is a mixed-precision
adaptation, not a claim of bitwise equivalence to the original FP32 runtime.
The single-frame start of a live session is explicitly marked as video, so the
upstream tracking object limit also applies to that first frame.
Because the public API only detects people, the actual model features for its
fixed text prompts are cached after their first evaluation. Text encoder weights
are then released from GPU memory. Other prompts are rejected, and sessions
receive separate feature copies. Original model weights and licenses remain in
the mounted model directory.
The Jetson API profile accepts one frame per append to bound the peak memory of
live video updates on the 8 GB device.
Before/after source hashes are preserved at `/opt/notices/jetson-patches.json`.
