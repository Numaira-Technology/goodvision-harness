"""本轮独立的离线文件源与完整capture记录；不调用额外模型或改变推理输出。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time

import av

from .video_to_stream import LiveFrame, probe_video
from .native_event_audit import append_file, ensure_directory

ROOT = Path('/run/output')


class OfflineCapture:
    def __init__(self, video_id):
        self.directory = ROOT / 'records' / video_id
        ensure_directory(self.directory)
        self.path = self.directory / 'offline-capture.jsonl'
        if self.path.exists():
            raise RuntimeError('offline capture already exists; never overwrite a run')
        self.lock = threading.Lock()
        self.events = {}

    def append(self, row):
        with self.lock:
            append_file(self.path, row)

    def observe_events(self, scene, session_id):
        # 与Ashley capture_live字段映射一致；记录更新，不通过轮询丢中间状态。
        for event in scene.get('scene', {}).get('events', []):
            key = str(event.get('event_id') or
                      f"{event['event_label']}:{event.get('frame_sequence')}")
            record = {
                **event,
                'event_label': str(event['event_label']),
                'subject_ids': list(event.get('subject_ids', ())),
                'first_evidence': event.get('first_evidence_timestamp_seconds'),
                'confirmed_since': event.get('confirmed_since_timestamp_seconds'),
                'ended_since': event.get('ended_since_timestamp_seconds'),
            }
            if self.events.get(key) != record:
                self.append({'kind': 'event', 'stream_session_id': session_id,
                             'event_id': key, 'event': record})
                self.events[key] = record


class OfflineVideoSource:
    realtime = False

    def __init__(self, path, runtime):
        self.path = Path(path)
        self.runtime = runtime
        self._stop = threading.Event()
        self.dropped_frames = 0
        self.capture = OfflineCapture(self.path.name)
        runtime._offline_capture = self.capture
        # 只对本离线源启用同PTS源帧；模型接纳仍使用原2fps时间槽。
        runtime._rich_labels.allow_equal_source_timestamps = True

    def stop(self):
        self._stop.set()

    @property
    def stopped(self):
        return self._stop.is_set()

    def _wait_for_capacity(self, timestamp):
        # 此处位于源读取线程，绝不能持有_state_lock等待内部提交。
        waiting_at = time.monotonic()
        while not self.stopped:
            runtime = self.runtime
            coordinator = runtime._rich_labels
            stats = coordinator.stats()
            if stats.get('state') in {'failed', 'closed', 'stopped'} or stats.get('recovery_required'):
                raise RuntimeError('offline tracking failed; source remains preserved')
            scene = runtime._scene
            if scene is None:
                raise RuntimeError('offline scene retired before file EOF')
            with scene._lock:
                pending_jobs = bool(scene._jobs)
                completed = scene._tracking_completed_timestamp
            worker = runtime._scene_worker
            worker_stats = worker.stats() if worker is not None else {}
            if worker_stats.get('state') in {'failed', 'stopped', 'stopping'}:
                raise RuntimeError('offline scene worker failed')
            if worker is not None:
                worker.wake()
            # 原native hotstart需要后续帧，允许4源秒有界前视，不要求逐帧发出。
            # 等待不会推进源时间，故不会因GPU/网络慢突破40源秒保留窗口。
            lag = timestamp - (completed if completed is not None else 0.0)
            busy = pending_jobs or worker_stats.get('active_requests', 0) or worker_stats.get('pending_completions', 0)
            if lag <= 4.0 and not busy:
                return
            for name in ('stop-loss.json', 'balance-blocked.json', 'cancel-requested.json'):
                if (ROOT / name).exists():
                    raise RuntimeError('offline controlled stop requested')
            if time.monotonic() - waiting_at > 360:
                raise RuntimeError('offline pipeline made no capacity progress for 360 seconds')
            self._stop.wait(0.02)

    def frames(self):
        first_pts = None
        last_timestamp = None
        count = 0
        repeated_timestamps = 0
        dts_fallback_frames = 0
        raw_identity = None
        file_info = None
        timestamp_basis = 'decoded_pts_or_decoder_dts_relative_to_first_frame'
        try:
            with av.open(str(self.path)) as container:
                # 无FRAME线程时，解码器frame.dts可提供缺PTS帧的呈现时间。
                # PyAV默认GENPTS保持历史行为；不能称返回值全是文件原始PTS。
                container.streams.video[0].codec_context.thread_type = 'SLICE'
                if 'avi' in container.format.name.split(','):
                    # AVI按仓库原VideoFileStream的时间口径，不依赖其不完整PTS。
                    # 只恢复时间规则；仍逐帧离线解码、背压，不恢复墙钟播放。
                    file_info = probe_video(self.path)
                    if not math.isfinite(file_info.source_fps) or file_info.source_fps <= 0:
                        raise ValueError('AVI file frame rate is invalid')
                    timestamp_basis = 'avi_original_frame_index_over_file_fps'
                self.capture.append({'kind': 'source_clock',
                                     'timestamp_basis': timestamp_basis,
                                     'container_format': container.format.name,
                                     'pyav_version': av.__version__,
                                     'container_flags': container.flags,
                                     'pyav_genpts_enabled': bool(container.flags & 1),
                                     'file_fps': file_info.source_fps if file_info else None,
                                     'declared_frame_count': file_info.frame_count if file_info else None,
                                     'wall_clock_pacing': False})
                for index, frame in enumerate(container.decode(video=0)):
                    if self.stopped:
                        return
                    raw_identity = {'frame': index, 'pts': frame.pts, 'dts': frame.dts,
                                    'time_base': str(frame.time_base)}
                    if file_info is not None:
                        timestamp = index / file_info.source_fps
                        self.capture.append({'kind': 'source_timestamp_from_frame_index',
                                             **raw_identity, 'timestamp_seconds': timestamp,
                                             'file_fps': file_info.source_fps})
                    else:
                        if frame.time_base is None:
                            raise ValueError('source frame has no time base')
                        tick = frame.pts
                        if tick is None:
                            tick = frame.dts
                            if tick is None:
                                raise ValueError('source frame has neither PTS nor decoder DTS')
                            dts_fallback_frames += 1
                            self.capture.append({'kind': 'source_timestamp_fallback',
                                                 **raw_identity, 'basis': 'decoder_dts_no_frame_threading'})
                        pts = float(tick * frame.time_base)
                        if first_pts is None:
                            first_pts = pts
                        timestamp = pts - first_pts
                    if not math.isfinite(timestamp) or timestamp < 0 or (
                        last_timestamp is not None and timestamp < last_timestamp
                    ):
                        raise ValueError('source presentation timestamps are invalid or decreasing')
                    if last_timestamp is not None and timestamp == last_timestamp:
                        repeated_timestamps += 1
                        self.capture.append({'kind': 'source_timestamp_repeat', **raw_identity,
                                             'timestamp_seconds': timestamp})
                    self._wait_for_capacity(timestamp)
                    if self.stopped:
                        return
                    yield LiveFrame(index=index, timestamp_seconds=timestamp,
                                    bgr=frame.to_ndarray(format='bgr24'))
                    count += 1
                    last_timestamp = timestamp
            if not count:
                raise ValueError('offline video decoded no frames')
            if file_info is not None and count != file_info.frame_count:
                raise ValueError('AVI decoded frame count differs from file metadata')
            self.capture.append({'kind': 'source_eof', 'decoded_frames': count,
                                 'first_pts_seconds': first_pts,
                                 'last_timestamp_seconds': last_timestamp,
                                 'timestamp_basis': timestamp_basis,
                                 'file_fps': file_info.source_fps if file_info else None,
                                 'declared_frame_count': file_info.frame_count if file_info else None,
                                 'dts_fallback_frames': dts_fallback_frames,
                                 'repeated_source_timestamps': repeated_timestamps,
                                 'wall_clock_pacing': False, 'dropped_due_to_lag': 0})
        except BaseException as error:
            self.capture.append({'kind': 'source_error', 'error_type': type(error).__name__,
                                 'detail': str(error), 'decoded_frames': count,
                                 'raw_identity': raw_identity,
                                 'previous_timestamp_seconds': last_timestamp})
            raise


def install_capture_hooks(runtime_class):
    original_batch = runtime_class._on_tracking_batch
    original_update = runtime_class._on_scene_update
    original_release = runtime_class._release_stopped_session

    def record_batch(self, batch):
        detector_native = getattr(batch, 'backend', None) == 'detector_stream'
        if detector_native and len(batch.detector_provenance) != len(batch.frames):
            raise RuntimeError('detector provenance is missing from a native frame')
        original_batch(self, batch)
        session = self._session
        sink = getattr(self, '_offline_capture', None)
        if sink is None or session is None or session.stream_session_id != batch.stream_session_id:
            return
        for index, frame in enumerate(batch.frames):
            mapped = session.person_registry.snapshot_at(frame.timestamp_seconds, frame.frame_sequence)
            if mapped is None or (mapped.timestamp_seconds, mapped.frame_sequence) != (frame.timestamp_seconds, frame.frame_sequence):
                raise RuntimeError('offline canonical capture lost exact source frame')
            sink.append({'kind': 'frame', 'stream_session_id': batch.stream_session_id,
                         'frame': {'t': mapped.timestamp_seconds,
                                   'frame': mapped.frame_sequence,
                                   'subjects': [{'id': obj.track_id,
                                                 'bbox': [obj.bbox.x1, obj.bbox.y1,
                                                          obj.bbox.x2, obj.bbox.y2],
                                                 **({'native_track': dict(obj.attributes['native_track']),
                                                     'observation_state': obj.observation_state}
                                                    if detector_native else {})}
                                                for obj in mapped.objects]},
                         **({'detector_model': batch.detector_model,
                             'detector_provenance': batch.detector_provenance[index],
                             'backend': batch.backend} if detector_native else {})})
        if self._scene is not None:
            sink.observe_events(self._scene.snapshot(include_history=True), batch.stream_session_id)

    def record_update(self):
        original_update(self)
        with self._state_lock:
            sink = getattr(self, '_offline_capture', None)
            scene = self._scene
            if sink is not None and scene is not None:
                sink.observe_events(scene.snapshot(include_history=True), scene.stream_session_id)

    def record_release(self, session):
        sink = getattr(session.video_stream, 'capture', None)
        if sink is not None and session is self._session:
            sink.append({'kind': 'tracking_finished',
                         'stream_session_id': session.stream_session_id,
                         'stats': self._rich_labels.stats()})
        return original_release(self, session)

    runtime_class._on_tracking_batch = record_batch
    runtime_class._on_scene_update = record_update
    runtime_class._release_stopped_session = record_release
