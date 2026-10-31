"""Fixed CUDA image encoders and paced twelve-slot video classification."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import queue
import statistics
import threading
import time

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from base_timing import TimedVideoFileStream


PROMPTS = {
    "fight": ["Two people are physically fighting, punching or kicking each other.",
              "People are walking, standing or talking peacefully."],
    "smoking": ["A person is smoking a cigarette.",
                "A person is eating food or drinking from a cup."],
    "fall": ["A person is losing balance and falling to the ground.",
             "A person is standing, walking or sitting steadily."],
    "weapon": ["A person is holding a gun or a knife.",
               "A person is holding an ordinary object such as a phone."],
    "bag_knife": ["A knife is visibly exposed inside a bag or being taken out of a bag.",
                  "A bag contains ordinary items with no visible knife."],
    "bag_alcohol_bottle": [
        "A bottle recognizable as an alcohol bottle is visibly exposed inside a bag or being taken out of a bag.",
        "A bag contains ordinary items with no visible alcohol bottle.",
    ],
}
REVISIONS = {
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
}
_CUDA_LOCK = threading.Lock()
_CUDA_USERS = 0
_SAVED_TF32 = None
_COUNTERS = ("opportunities", "completed", "busy_missed", "late_missed", "local_failures")


def _features(value):
    tensor = value.pooler_output if hasattr(value, "pooler_output") else value
    return F.normalize(tensor.float(), dim=-1)


def _save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    return {"n": len(values), "p50": values[math.ceil(len(values) * .5) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1],
            "mean": statistics.mean(values), "min": values[0], "max": values[-1]}


class ClipEncoder:
    """Keep one fixed FP32 CUDA checkpoint and its text vectors loaded."""

    def __init__(self, model_id, cache_dir):
        global _CUDA_USERS, _SAVED_TF32
        revision = REVISIONS[model_id]
        self.model = self.processor = self.text_vectors = None
        self._closed = False
        with _CUDA_LOCK:
            if _CUDA_USERS == 0:
                _SAVED_TF32 = (torch.backends.cuda.matmul.allow_tf32,
                               torch.backends.cudnn.allow_tf32)
            _CUDA_USERS += 1
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        started = time.monotonic()
        try:
            load = {"revision": revision, "cache_dir": str(cache_dir)}
            self.processor = AutoProcessor.from_pretrained(model_id, **load)
            self.model = AutoModel.from_pretrained(
                model_id, **load, dtype=torch.float32,
            ).eval().to("cuda")
            siglip = "siglip" in model_id
            texts = [text for pair in PROMPTS.values() for text in pair]
            if siglip:
                texts = [text.lower() for text in texts]
            with torch.inference_mode():
                tokens = self.processor(
                    text=texts, padding="max_length" if siglip else True,
                    max_length=64 if siglip else 77, truncation=True, return_tensors="pt",
                )
                tokens = {key: value.to("cuda") for key, value in tokens.items()}
                self.text_vectors = _features(self.model.get_text_features(**tokens))
            del tokens
            torch.cuda.synchronize()
            self.metadata = {
                "model": model_id, "actual_model": model_id, "revision": revision,
                "device": "cuda", "gpu": torch.cuda.get_device_name(0),
                "precision": "float32", "tf32": False, "window_frames": 1,
                "cpu_threads": torch.get_num_threads(), "prompts": PROMPTS,
                "score": "positive_cosine - contrast_cosine",
                "decision": "margin > threshold", "ground_truth_input": False,
                "load_and_text_seconds": time.monotonic() - started,
            }
        except BaseException:
            self.close()
            raise

    def prepare(self, image):
        """Apply the checkpoint's official image processor on the CPU."""
        return self.processor(images=image, return_tensors="pt")

    def predict(self, prepared):
        """Return six positive/contrast cosine pairs after CUDA readback."""
        with torch.inference_mode():
            pixels = {key: value.to("cuda") for key, value in prepared.items()}
            vectors = _features(self.model.get_image_features(**pixels))
            scores = (vectors @ self.text_vectors.T).cpu().reshape(len(PROMPTS), 2)
        predictions = {
            label: {"positive_cosine": float(scores[index, 0]),
                    "contrast_cosine": float(scores[index, 1]),
                    "margin": float(scores[index, 0] - scores[index, 1])}
            for index, label in enumerate(PROMPTS)
        }
        if any(not math.isfinite(value) for item in predictions.values() for value in item.values()):
            raise ValueError("non-finite cosine score")
        return predictions

    def close(self):
        global _CUDA_USERS, _SAVED_TF32
        if self._closed:
            return
        self._closed = True
        self.model = self.processor = self.text_vectors = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        finally:
            with _CUDA_LOCK:
                _CUDA_USERS -= 1
                if _CUDA_USERS == 0 and _SAVED_TF32 is not None:
                    (torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32) = _SAVED_TF32
                    _SAVED_TF32 = None


def run_streams(media, encoder, thresholds, output, slots=12, interval=.8):
    """Read each video once for quality, with explicitly marked tail load."""
    if type(slots) is not int or not 1 <= slots <= 12:
        raise ValueError("slots must be an integer from one to twelve")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be positive and finite")
    names = [entry["filename"] for entry in media]
    if len(names) != len(set(names)):
        raise ValueError("each primary filename must be unique")
    selected_thresholds = {label: float(thresholds.get(label, 0.0)) for label in PROMPTS}
    if any(not math.isfinite(value) for value in selected_thresholds.values()):
        raise ValueError("thresholds must be finite")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = (output / "clip-records.jsonl").open("x", encoding="utf-8")
    stop, condition, io_lock = threading.Event(), threading.Condition(), threading.Lock()
    pending = queue.Queue(maxsize=slots)
    todo, reusable = deque(media), []
    states, shared_failure = [], []
    inflight = {slot: False for slot in range(1, slots + 1)}
    remaining = len(media)
    filler_number = 0
    times = {scope: {key: [] for key in ("input_to_result_seconds", "cpu_prepare_seconds",
                                        "queue_seconds", "gpu_and_readback_seconds")}
             for scope in ("primary", "all")}
    started = time.monotonic()
    _save(output / "run.json", {
        **encoder.metadata, "slots": slots, "interval_seconds": interval,
        "sampling_fps": 1 / interval, "thresholds": selected_thresholds,
        "planned_primary_videos": len(media),
        "wall_time_boundary": "sampled frame actual input to classification result ready",
        "wall_time_excludes": ["frontend publication", "natural event onset"],
        "tail_load": "Completed primary videos may fill idle slots while other primary videos remain; transitions may have gaps.",
    })

    def record(row):
        with io_lock:
            try:
                records.write(json.dumps(row, allow_nan=False) + "\n")
                records.flush()
            except Exception as error:
                shared_failure.append("recording_failed:" + str(error))
                stop.set()
                raise

    def choose(slot):
        nonlocal filler_number
        with condition:
            while not stop.is_set() and remaining:
                if todo:
                    entry, primary = todo.popleft(), True
                    source_id = entry["filename"]
                elif reusable:
                    filler_number += 1
                    entry, primary = reusable[(filler_number - 1) % len(reusable)], False
                    source_id = f"fill-{slot}-{filler_number}"
                else:
                    condition.wait(.05)
                    continue
                state = {
                    "source_id": source_id, "filename": entry["filename"],
                    "quality_reference": primary, "slot": slot,
                    **dict.fromkeys(_COUNTERS, 0), "sample_complete": False,
                    "events_drained": False, "execution_complete": False,
                    "error": None, "start_monotonic": None, "end_monotonic": None,
                }
                states.append(state)
                return entry, state
        return None

    def wait_slot(slot):
        with condition:
            while inflight[slot] and not stop.is_set():
                condition.wait(.05)

    def reader(slot):
        nonlocal remaining
        while not stop.is_set():
            chosen = choose(slot)
            if chosen is None:
                return
            entry, state = chosen
            stream = iterator = None
            clock, next_sample = {}, 0.0
            try:
                stream = TimedVideoFileStream(
                    str(entry["video_path"]), start_seconds=0, stop=stop,
                    input_observer=clock.update, repeat=False,
                )
                state["start_monotonic"] = time.monotonic()
                iterator = stream.frames(origin=state["start_monotonic"])
                for frame in iterator:
                    if stop.is_set() or (not state["quality_reference"] and remaining == 0):
                        state["stopped_after_primary"] = not stop.is_set()
                        break
                    if frame.timestamp_seconds + 1e-8 < next_sample:
                        continue
                    elapsed = math.floor((frame.timestamp_seconds - next_sample) / interval + 1e-8) + 1
                    next_sample += elapsed * interval
                    with condition:
                        state["opportunities"] += elapsed
                        state["late_missed"] += elapsed - 1
                        if inflight[slot]:
                            state["busy_missed"] += 1
                            continue
                        inflight[slot] = True
                    row = {
                        "source_id": state["source_id"], "filename": entry["filename"],
                        "quality_reference": state["quality_reference"], "slot": slot,
                        "source_times": [frame.timestamp_seconds],
                        "frame_index": clock["raw_frame_index"],
                        "timestamp_seconds": frame.timestamp_seconds,
                        "last_frame_ingress": clock["ingress_monotonic"],
                    }
                    prepare_started = time.monotonic()
                    try:
                        image = Image.fromarray(frame.bgr[:, :, ::-1].copy())
                        prepared = encoder.prepare(image)
                        row["cpu_prepare_seconds"] = time.monotonic() - prepare_started
                        row["queued_monotonic"] = time.monotonic()
                        pending.put_nowait((state, row, prepared))
                    except Exception as error:
                        row["cpu_prepare_seconds"] = time.monotonic() - prepare_started
                        with condition:
                            state["local_failures"] += 1
                            inflight[slot] = False
                            condition.notify_all()
                        record(row | {"classification_ready": None, "readable": None,
                                      "predictions": None, "errors": str(error)})
                state["sample_complete"] = bool(
                    not stop.is_set() and stream.eof_seen and stream.last_raw_index is not None
                    and stream.last_raw_index >= stream.info.frame_count - 1
                )
                if not state["sample_complete"] and not state.get("stopped_after_primary"):
                    state["error"] = "source_incomplete"
            except Exception as error:
                state["error"] = "source_failed:" + str(error)
            finally:
                try:
                    if iterator is not None:
                        iterator.close()
                except Exception as error:
                    state["error"] = state["error"] or "source_close_failed:" + str(error)
                try:
                    if stream is not None:
                        stream.stop()
                        state["input_statistics"] = dict(stream.input_timing_statistics)
                except Exception as error:
                    state["error"] = state["error"] or "source_stop_failed:" + str(error)
                if state["quality_reference"] and stream is not None and stream.last_raw_index is not None:
                    source_fps = stream.info.source_fps
                    if math.isfinite(source_fps) and source_fps > 0:
                        due = math.floor((stream.last_raw_index / source_fps) / interval + 1e-8) + 1
                        with condition:
                            missed_tail = max(0, due - state["opportunities"])
                            state["opportunities"] += missed_tail
                            state["late_missed"] += missed_tail
                state["input_end_monotonic"] = time.monotonic()
                wait_slot(slot)
                state["end_monotonic"] = time.monotonic()
                with condition:
                    state["events_drained"] = not inflight[slot]
                    state["execution_complete"] = bool(
                        state["sample_complete"] and state["events_drained"] and state["error"] is None
                    )
                    if state["quality_reference"]:
                        remaining -= 1
                        if state["sample_complete"]:
                            reusable.append(entry)
                    condition.notify_all()

    try:
        with ThreadPoolExecutor(max_workers=slots, thread_name_prefix="clip-slot") as pool:
            futures = []
            try:
                for slot in range(1, slots + 1):
                    futures.append(pool.submit(reader, slot))
                while any(not future.done() for future in futures) or not pending.empty():
                    if stop.is_set():
                        break
                    try:
                        state, row, prepared = pending.get(timeout=.02)
                    except queue.Empty:
                        continue
                    try:
                        gpu_started = time.monotonic()
                        try:
                            predictions = encoder.predict(prepared)
                            for label, item in predictions.items():
                                item["positive"] = item["margin"] > selected_thresholds[label]
                            ready = time.monotonic()
                            result = row | {
                                "classification_ready": ready, "readable": ready,
                                "queue_seconds": gpu_started - row["queued_monotonic"],
                                "gpu_and_readback_seconds": ready - gpu_started,
                                "input_to_result_seconds": ready - row["last_frame_ingress"],
                                "predictions": predictions, "errors": None,
                            }
                        except Exception as error:
                            result = row | {"classification_ready": None, "readable": None,
                                            "predictions": None, "errors": str(error)}
                        record(result)
                        with condition:
                            state["completed" if result["errors"] is None else "local_failures"] += 1
                            if result["errors"] is None:
                                scopes = ("primary", "all") if state["quality_reference"] else ("all",)
                                for scope in scopes:
                                    for key in times[scope]:
                                        times[scope][key].append(result[key])
                    finally:
                        prepared = None
                        with condition:
                            inflight[state["slot"]] = False
                            condition.notify_all()
                        pending.task_done()
                for future in futures:
                    future.result()
            finally:
                stop.set()
                with condition:
                    condition.notify_all()
    except Exception as error:
        shared_failure.append("runtime_failed:" + str(error))
    finally:
        stop.set()
        records.close()

    finished = time.monotonic()
    primary = [state for state in states if state["quality_reference"]]
    intervals = [
        {key: state.get(key) for key in ("source_id", "slot", "quality_reference",
                                        "start_monotonic", "input_end_monotonic", "end_monotonic")}
        for state in states
    ]
    summary = {
        "execution_complete": len(primary) == len(media)
        and all(state["execution_complete"] for state in primary) and not shared_failure,
        "shared_failure": shared_failure or None, "planned_primary_videos": len(media),
        "primary_sources_started": len(primary), "primary_sources_remaining": remaining,
        "started_monotonic": started, "finished_monotonic": finished,
        "wall_seconds": finished - started, "slots": slots, "interval_seconds": interval,
        "thresholds": selected_thresholds, "streams": states,
        "active_slot_intervals": intervals,
        "wall_time_boundary": "sampled frame actual input to classification result ready",
        "wall_time_excludes": ["frontend publication", "natural event onset"],
    }
    for scope, members in (("primary", primary), ("all", states)):
        counters = {key: sum(state[key] for state in members) for key in _COUNTERS}
        counters["completed_fraction"] = (counters["completed"] / counters["opportunities"]
                                           if counters["opportunities"] else None)
        counters["source_count"] = len(members)
        counters["latency"] = {key: _distribution(values) for key, values in times[scope].items()}
        summary[scope] = counters
    _save(output / "completion.json", summary)
    return summary

