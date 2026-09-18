"""Apply audited low-memory loading and mixed-precision changes for Jetson."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise RuntimeError("Pinned source changed; review the Jetson loading patch")
    return text.replace(before, after, 1)


def patch_builder(text: str) -> str:
    # The final merged checkpoint supplies every tracker/detector parameter and
    # the worker still verifies all keys, shapes and values and reloads strictly.
    # Avoid a redundant partial load and GPU allocation of a discarded backbone.
    header_start = text.index("def build_sam3_multiplex_video_model(")
    header_end = text.index("\n):", header_start)
    header = text[header_start:header_end]
    text = replace_once(
        text,
        header,
        replace_once(
            header, "    compile=False,", "    compile=False,\n    build_backbone=True,"
        ),
    )
    unused_backbone = (
        "    tri_neck = _create_multiplex_tri_backbone(\n"
        '        compile_mode="max-autotune" if compile else None\n'
        "    )\n"
        "    backbone = TriHeadVisionOnly(\n"
        "        visual=tri_neck,\n"
        "        n_features=256,\n"
        "        scalp=0,\n"
        "    )"
    )
    text = replace_once(
        text,
        unused_backbone,
        "    backbone = None\n    if build_backbone:\n"
        + "\n".join("    " + line for line in unused_backbone.splitlines()),
    )
    text = replace_once(
        text,
        "tracker_model = build_sam3_multiplex_video_model(\n"
        "        checkpoint_path=checkpoint_path,\n        load_from_HF=False,",
        "tracker_model = build_sam3_multiplex_video_model(\n"
        '        checkpoint_path=None,\n        device="meta",\n'
        "        build_backbone=False,\n"
        "        load_from_HF=False,",
    )
    # Compilation is disabled. Compute positional caches only for resolutions
    # actually requested, instead of preallocating every supported GPU pyramid.
    if text.count("precompute_resolution=1008") != 4:
        raise RuntimeError("Pinned positional cache factories changed")
    text = text.replace("precompute_resolution=1008", "precompute_resolution=None")
    start = text.index(
        "    # Build tracker\n", text.index("def build_sam3_multiplex_video_predictor(")
    )
    end = text.index("    # Load checkpoint (auto-download", start)
    construction = text[start:end]
    text = (
        text[:start]
        + '    with torch.device("meta"):\n'
        + "".join(
            "    " + line if line.strip() else line
            for line in construction.splitlines(keepends=True)
        )
        + text[end:]
    )
    text = replace_once(
        text,
        'ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)',
        'ckpt = torch.load(checkpoint_path, map_location="cpu", '
        "weights_only=True, mmap=True)",
    )
    text = replace_once(
        text,
        "demo_model.load_state_dict(ckpt, strict=False)",
        "demo_model.load_state_dict(ckpt, strict=True, assign=True)",
    )
    gpu_transfer = """    _fp32_parameters = set()
    for _module in demo_model.modules():
        if (_module.__class__.__module__ == "sam3.model.decoder"
                and _module.__class__.__name__ == "TransformerDecoderLayer"):
            for _layer in (_module.linear1, _module.linear2):
                _fp32_parameters.update(id(_p) for _p in _layer.parameters())
    if not _fp32_parameters:
        raise RuntimeError("Expected full-precision decoder layers missing")
    _parameters, _buffers = {}, {}
    for _module in demo_model.modules():
        for _name, _parameter in list(_module._parameters.items()):
            if _parameter is None:
                continue
            _key = id(_parameter)
            if _key not in _parameters:
                _dtype = (
                    torch.bfloat16
                    if (_parameter.is_floating_point() and _parameter.ndim >= 2
                        and _key not in _fp32_parameters)
                    else _parameter.dtype
                )
                _parameters[_key] = torch.nn.Parameter(
                    _parameter.to(device="cuda", dtype=_dtype),
                    requires_grad=_parameter.requires_grad,
                )
            _module._parameters[_name] = _parameters[_key]
        for _name, _buffer in list(_module._buffers.items()):
            if _buffer is None:
                continue
            _key = id(_buffer)
            if _key not in _buffers:
                _buffers[_key] = _buffer.to(device="cuda")
            _module._buffers[_name] = _buffers[_key]
    del _parameters, _buffers, _fp32_parameters
    demo_model.eval()
    print("Contour: BF16 tensors; FP32 decoder FFN and original buffers", flush=True)
"""
    return replace_once(
        text,
        "    demo_model.cuda().eval()",
        "    del ckpt\n"
        "    # Recreate the sole nonpersistent meta buffer using upstream code.\n"
        "    for _module in demo_model.modules():\n"
        "        for _name, _buffer in _module.named_buffers(recurse=False):\n"
        "            if _buffer.is_meta:\n"
        '                if _name == "attn_mask" and _name in '
        "_module._non_persistent_buffers_set:\n"
        "                    _module.attn_mask = _module.build_causal_mask()\n"
        "                else:\n"
        '                    raise RuntimeError("Unmaterialized model buffer: " '
        "+ _name)\n"
        "    if any(_p.is_meta for _p in demo_model.parameters()):\n"
        '        raise RuntimeError("Unmaterialized model parameters")\n'
        "    import ctypes as _contour_ctypes\n"
        '    _contour_ctypes.CDLL("libc.so.6").malloc_trim(0)\n'
        '    print("Contour: mapped weights assigned; transferring to GPU", '
        "flush=True)\n"
        + gpu_transfer
        + '    print("Contour: GPU weight transfer completed", flush=True)',
    )


def patch_worker(text: str) -> str:
    text = replace_once(
        text,
        'raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)',
        'raw = torch.load(str(checkpoint), map_location="cpu", '
        "weights_only=True, mmap=True)",
    )
    text = replace_once(
        text,
        '            raise RuntimeError("CUDA GPU unavailable")',
        '            raise RuntimeError("CUDA GPU unavailable")\n'
        "        # CUDA allocations share physical RAM with the Jetson host.\n"
        "        # Fail this process before consuming all shared GPU memory.\n"
        "        torch.cuda.set_per_process_memory_fraction(0.55, device=0)\n"
        '        Path("/tmp/cache/torch/kernels").mkdir(parents=True, exist_ok=True)\n'
        '        print("Contour: CUDA allocation budget limited to 55%", flush=True)',
    )
    text = replace_once(
        text,
        "        del verified, state, raw",
        "        del verified, state, raw\n"
        "        for parameter in model.parameters():\n"
        "            if parameter.is_floating_point() and not "
        "torch.isfinite(parameter).all().item():\n"
        '                raise RuntimeError("Nonfinite runtime model parameters")\n'
        "        import gc, ctypes\n"
        "        gc.collect()\n"
        '        ctypes.CDLL("libc.so.6").malloc_trim(0)\n'
        "        torch.cuda.empty_cache()\n"
        "        from jetson_runtime import cache_person_text\n"
        "        cache_person_text(model.detector.backbone, torch)\n"
        '        print("Contour: strict checkpoint validation completed", flush=True)',
    )
    text = replace_once(
        text,
        "            for image in blanks:\n                image.close()",
        "            for image in blanks:\n                image.close()\n"
        "            gc.collect()\n"
        "            torch.cuda.empty_cache()\n"
        '            ctypes.CDLL("libc.so.6").malloc_trim(0)\n'
        '            print("Contour: after warmup allocated/reserved MiB",\n'
        "                  torch.cuda.memory_allocated() // 1048576,\n"
        "                  torch.cuda.memory_reserved() // 1048576, flush=True)",
    )
    return replace_once(
        text,
        'self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)',
        'self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16, '
        "cache_enabled=False)",
    )


def patch_live(text: str) -> str:
    # Autocast's cached lower-precision weights would coexist with the complete
    # FP32 model. Retain per-operation autocast without keeping a second copy.
    text = replace_once(
        text,
        "    def run(self, session, images, *, final):",
        "    def discard(self, session):\n"
        "        super().discard(session)\n"
        "        import gc, ctypes\n"
        "        gc.collect()\n"
        "        self.video.backend._torch.cuda.empty_cache()\n"
        '        ctypes.CDLL("libc.so.6").malloc_trim(0)\n\n'
        "    def run(self, session, images, *, final):",
    )
    text = replace_once(
        text,
        "                session.native_state = state",
        '                state["is_image_only"] = False\n'
        "                session.native_state = state",
    )
    text = replace_once(
        text,
        'native._torch.autocast(device_type="cuda", dtype=native._torch.bfloat16)',
        'native._torch.autocast(device_type="cuda", dtype=native._torch.bfloat16, '
        "cache_enabled=False)",
    )
    text = replace_once(
        text,
        "            native._torch.cuda.synchronize()",
        "            native._torch.cuda.synchronize()\n"
        "            # Return unused GPU workspace to the shared-memory host.\n"
        "            native._torch.cuda.empty_cache()",
    )
    return replace_once(
        text,
        "        self.retention = BoundedRetention(self.model, "
        "self.settings.history_frames)",
        "        self.retention = BoundedRetention(self.model, "
        "self.settings.history_frames)\n"
        '        print("Contour: live start allocated/reserved MiB",\n'
        "              self.video.backend._torch.cuda.memory_allocated() // 1048576,\n"
        "              self.video.backend._torch.cuda.memory_reserved() // 1048576,\n"
        "              flush=True)",
    )


def patch_vit(text: str) -> str:
    # Drop-path rates are tiny Python configuration scalars, not model weights.
    return replace_once(
        text,
        "torch.linspace(0, drop_path_rate, depth)",
        'torch.linspace(0, drop_path_rate, depth, device="cpu")',
    )


def patch_decoder(text: str) -> str:
    # Upstream explicitly disables autocast for this numerically sensitive FFN.
    # Its weights stay FP32; promote incoming mixed-precision activations too.
    return replace_once(
        text,
        '        with torch.amp.autocast(device_type="cuda", enabled=False):\n'
        "            tgt2 = self.linear2("
        "self.dropout3(self.activation(self.linear1(tgt))))",
        '        with torch.amp.autocast(device_type="cuda", enabled=False):\n'
        "            tgt2 = self.linear2(\n"
        "                self.dropout3(self.activation(self.linear1(tgt.float())))\n"
        "            )",
    )


def patch_routes(text: str) -> str:
    return replace_once(
        text,
        'SessionProfile("sam_live", "sam3.1_video")',
        'SessionProfile("sam_live", "sam3.1_video", max_append_frames=1)',
    )


def main():
    distribution = importlib.metadata.distribution("sam3")
    origin = json.loads(distribution.read_text("direct_url.json"))
    revision = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
    if origin.get("vcs_info", {}).get("commit_id") != revision:
        raise RuntimeError("Jetson patches require the approved source revision")
    builder = Path(distribution.locate_file("sam3/model_builder.py"))
    changes = []
    for path, transform in (
        (builder, patch_builder),
        (Path("/engine/sam31.py"), patch_worker),
        (Path("/engine/sam_live.py"), patch_live),
        (Path("/engine/perception_routes.py"), patch_routes),
        (Path(distribution.locate_file("sam3/model/vitdet.py")), patch_vit),
        (Path(distribution.locate_file("sam3/model/decoder.py")), patch_decoder),
    ):
        before = path.read_bytes()
        after = transform(before.decode("utf-8")).encode("utf-8")
        path.write_bytes(after)
        changes.append(
            {
                "path": str(path),
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "after_sha256": hashlib.sha256(after).hexdigest(),
            }
        )
    Path("/opt/notices/jetson-patches.json").write_text(
        json.dumps({"upstream_revision": revision, "changes": changes}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        "Applied Jetson BF16 runtime patches; original checkpoint validation retained"
    )


if __name__ == "__main__":
    main()
