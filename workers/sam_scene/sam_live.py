"""Causal SAM plugin: persistent learned tracking, arrived-frame outputs only.

The pinned Prod's detector/tracker is unchanged. Its future-output confirmation
buffer is replaced by causal postprocessing. Native three-detection confirmation
is reported explicitly; provisional masks are never called confirmed evidence.
"""

from __future__ import annotations

import time

if __package__:
    from .continuous import NativeContinuation, Session
    from .streaming_state import BoundedRetention, make_prod_model
else:
    from continuous import NativeContinuation, Session
    from streaming_state import BoundedRetention, make_prod_model


class CausalPostprocessing:
    """Specialize only forward delivery, retaining native rejection heuristics."""

    def initialize_causal_prompt(self, state):
        # Upstream classifies a one-frame list as an image. This is the first
        # frame of a continuing video: keep video tracking and object limits.
        state["is_image_only"] = False
        # The pinned add_prompt also infers frame 0. Re-running propagation at 0
        # would count the same observation twice towards detector confirmation.
        # Install only its audited text/backbone initialization here; the normal
        # causal propagation performs the first and only update of source 0.
        self.reset_state(state)
        state["text_prompt"] = "person"
        state["input_batch"].find_text_batch[0] = "person"
        for stage in state["input_batch"].find_inputs:
            stage.text_ids[...] = self.TEXT_ID_FOR_TEXT
        state["backbone_out"] = self._init_backbone_out(state)

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        is_last_batch=False,
    ):
        if reverse:
            raise ValueError("causal SAM only accepts forward arrived frames")
        self._compile_model()
        order, _ = self._get_processing_order(
            inference_state, start_frame_idx, max_frame_num_to_track, False
        )
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }
        generator = inference_state["generator_state"]
        if generator["hotstart_buffer"] or generator["postprocess_yield_list"]:
            raise RuntimeError("causal SAM cannot inherit buffered output state")
        observations = inference_state.setdefault("causal_observations", {})
        observations.clear()
        for index in order:
            raw = self._run_single_frame_inference(inference_state, index, False)
            removed = set(raw["removed_obj_ids"])
            provisional = set(raw["unconfirmed_obj_ids"])
            suppressed = raw["suppressed_obj_ids"]
            generator["unconfirmed_obj_ids_per_frame"][index] = provisional
            # Cache and emit the current real observation. Never look at index+2
            # or revise a past output using later detector confirmation.
            self._cache_frame_outputs(
                inference_state,
                index,
                raw["obj_id_to_mask"],
                suppressed_obj_ids=suppressed,
                removed_obj_ids=removed,
                unconfirmed_obj_ids=None,
            )
            output = self._postprocess_output(
                inference_state, raw, removed, suppressed, None
            )
            visible = {int(value) for value in output["out_obj_ids"]}
            observations[index] = [
                str(value) for value in sorted(visible.intersection(provisional))
            ]
            yield index, output


class SamLiveContinuation(NativeContinuation):
    """A separately selectable output profile over shared verified model weights."""

    def warmup(self):
        if not self.video.gpu_ready:
            raise RuntimeError("native CUDA video model is unavailable")
        self.model = make_prod_model(
            self.video.backend,
            hotstart_delay=5,
            postprocess_batch_size=1,
            model_mixin=CausalPostprocessing,
        )
        if not self.model.masklet_confirmation_enable:
            raise RuntimeError("causal SAM requires native confirmation metadata")
        self.retention = BoundedRetention(self.model, self.settings.history_frames)
        from PIL import Image

        # Exercise startup1, multi-frame append, ring rollover, pruning and empty
        # EOF on actual CUDA before claiming this specific route is ready.
        session = Session("live-warmup", "live-warmup", "live-warmup", time.monotonic())
        seen = []
        try:
            for count in (1, 4, 4, 4, 4, 4, 4, 4, 4):
                images = [Image.new("RGB", (320, 192)) for _ in range(count)]
                try:
                    arrived = list(self.run(session, images, final=False))
                    indices = [index for index, _ in arrived]
                    if indices != list(
                        range(session.total_frames, session.total_frames + count)
                    ):
                        raise RuntimeError("causal native warmup omitted source frames")
                    seen.extend(indices)
                    session.total_frames += count
                    session.emitted_index = seen[-1]
                    self.prune(session)
                finally:
                    for image in images:
                        image.close()
            if list(self.run(session, [], final=True)) or seen != list(range(33)):
                raise RuntimeError("causal native EOF invented source frames")
            self.video.backend._torch.cuda.synchronize()
            self.gpu_ready = True
        finally:
            self.discard(session)

    def run(self, session, images, *, final):
        native = self.video.backend
        with (
            native._torch.inference_mode(),
            native._torch.autocast(device_type="cuda", dtype=native._torch.bfloat16),
        ):
            state = session.native_state
            if state is None:
                if not images:
                    raise RuntimeError("causal SAM requires an arrived first image")
                state = self.model.init_state(
                    resource_path=list(images),
                    offload_video_to_cpu=True,
                    async_loading_frames=False,
                )
                session.native_state = state
                self.model.initialize_causal_prompt(state)
                self.retention.install(state, allow_partial=True)
                session.native_retention_installed = True
            elif images:
                self.retention.append(
                    self.model,
                    state,
                    images,
                    frame_limit=self.settings.max_session_frames,
                )
            if images:
                yield from self.model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=session.total_frames,
                    max_frame_num_to_track=len(images),
                    reverse=False,
                    is_last_batch=final,
                )
            else:
                state.setdefault("causal_observations", {}).clear()
            if (
                state["tracker_metadata"].get("max_obj_id", -1)
                >= self.settings.max_cumulative_native_ids
            ):
                from fastapi import HTTPException

                raise HTTPException(409, "reset_required")
            native._torch.cuda.synchronize()

    def route_metadata(self, session, snapshots):
        observations = (session.native_state or {}).get("causal_observations", {})
        first = session.emitted_index - len(snapshots) + 1
        return {
            "confirmation_policy": "causal_native_three_detections",
            "observations": [
                {
                    "frame_sequence": snapshot["frame_sequence"],
                    "provisional_object_ids": [
                        item["attributes"]["native_track"]["object_id"]
                        for item in snapshot["objects"]
                        if item["attributes"]["native_track"]["object_id"]
                        in observations.get(first + offset, [])
                    ],
                }
                for offset, snapshot in enumerate(snapshots)
            ],
        }
