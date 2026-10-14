"""旁路保存原生窗口、accepted 证据和生命周期；不改返回值或调度。"""
from __future__ import annotations

import hashlib
import copy
import json
import os
import threading
import time
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

ROOT = Path(os.environ.get("ASHLEY_OUTPUT_ROOT", "/run/output"))
LOCK = threading.RLock()
ERRORS = 0
INSTALLED = False
IO_FAILED = threading.Event()
AUDIT_IO_EXIT_CODE = 74


def fail_audit_io(path, error):
    """审计不可写时立即停止整个服务；即使哨兵也写不成仍用退出码通知父控。"""
    IO_FAILED.set()
    row = {"reason": "GLOBAL_AUDIT_IO_FAILURE", "path": str(path),
           "error_type": type(error).__name__, "errno": getattr(error, "errno", None),
           "time_epoch": time.time(), "exit_code": AUDIT_IO_EXIT_CODE}
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    # 不调用普通审计函数，避免文件系统故障递归；stderr不含请求正文或凭据。
    for target in (ROOT / "audit-io-failed.json", ROOT / "stop-loss.json"):
        try:
            with target.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            pass
    try:
        os.write(2, payload)
    except OSError:
        pass
    # worker会吞普通Exception；退出才能阻止其他并发线程继续发起付费请求。
    # 父控制器看到专用74退出码后全局退出，由其finally释放本任务资源。
    os._exit(AUDIT_IO_EXIT_CODE)


def require_audit_writable():
    if IO_FAILED.is_set():
        raise RuntimeError("GLOBAL_AUDIT_IO_FAILURE: audit output is unavailable")


def ensure_directory(path):
    require_audit_writable()
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError as error:
        fail_audit_io(path, error)


def write_json_file(path, row):
    require_audit_writable()
    try:
        Path(path).write_text(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as error:
        fail_audit_io(path, error)


def write_stop_loss(row, *, global_failure=False):
    """普通停止只保留首次原因；发布全局错误可升级，IO故障始终优先。"""
    require_audit_writable()
    path = ROOT / 'stop-loss.json'
    with LOCK:
        if path.exists():
            if not global_failure:
                return
            try:
                prior = json.loads(path.read_text())
            except OSError as error:
                fail_audit_io(path, error)
            if prior.get('reason') == 'GLOBAL_AUDIT_IO_FAILURE':
                return
        write_json_file(path, row)


def append_file(path, row):
    require_audit_writable()
    with LOCK:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with Path(path).open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            fail_audit_io(path, error)


def append(name, row):
    append_file(ROOT / name, row)


def no_images(value):
    if isinstance(value, dict):
        return {key: no_images(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [no_images(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_data_url_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "data_url_characters": len(value)}
    return value


def install():
    global INSTALLED
    if INSTALLED:
        return
    from .scene_model import TemporalVLMInspector
    from .scene_runtime import SceneRuntime
    from .scene_state import SceneState
    from .scene_results import SceneResultPublisher

    baseline_path = Path(os.environ['ASHLEY_BASELINE_RESPONSE_IDS'])
    baseline_values = json.loads(baseline_path.read_text())
    if (not isinstance(baseline_values, list)
            or any(not isinstance(value, str) or not value for value in baseline_values)):
        raise RuntimeError('GLOBAL_INVALID_BASELINE_RESPONSE_IDS')
    baseline_response_ids = set(baseline_values)
    try:
        known_cost_limit = Decimal(os.environ['ASHLEY_KNOWN_COST_LIMIT_USD'])
    except (KeyError, InvalidOperation) as error:
        raise RuntimeError('GLOBAL_INVALID_KNOWN_COST_LIMIT') from error
    if known_cost_limit != Decimal('20'):
        raise RuntimeError('GLOBAL_INVALID_KNOWN_COST_LIMIT')
    dispatch_reserve = Decimal('0.05')

    def receipt_cost(response):
        usage = response.get('usage') if isinstance(response, dict) else None
        value = usage.get('cost') if isinstance(usage, dict) else None
        try:
            amount = Decimal(str(value))
        except InvalidOperation as error:
            raise RuntimeError('GLOBAL_MISSING_RESPONSE_COST') from error
        if not amount.is_finite() or amount < 0:
            raise RuntimeError('GLOBAL_INVALID_RESPONSE_COST')
        return amount

    # 只复用完整请求相同的成功回执，图片以原始 data URL 哈希比较；不改写人物 ID。
    previous = Path(os.environ['ASHLEY_PREVIOUS_RESPONSES'])
    cache = {}
    counts = {}
    cache_sources = {}
    blocked = set()
    in_flight = set()
    seen_calls = {}
    legacy_receipts = set()
    def request_hash(request):
        return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'),
                                        ensure_ascii=False).encode()).hexdigest()
    for line in previous.read_text().splitlines():
        if not line.strip():
            continue
        saved = json.loads(line)
        digest = request_hash(saved['request'])
        if saved.get('request_sha256', digest) != digest:
            raise RuntimeError('GLOBAL_PREVIOUS_REQUEST_HASH_MISMATCH')
        call_id = saved.get('call_id')
        if call_id in seen_calls:
            if seen_calls[call_id] != (digest, saved['request']):
                raise RuntimeError('GLOBAL_PREVIOUS_CALL_ID_MISMATCH')
            continue
        if call_id:
            seen_calls[call_id] = (digest, saved['request'])
        counts[digest] = counts.get(digest, 0) + 1
        response_id = saved.get('response', {}).get('id')
        if response_id:
            cache.setdefault(digest, saved['response'])
            cache_sources[digest] = str(previous)
            if not call_id:
                legacy_receipts.add((digest, response_id))
        elif saved.get('error', {}).get('status_code') not in (402, 429):
            # 旧未知只封锁相同完整输入，不阻断其他独立题的启动。
            blocked.add(digest)
    # output可含父控制器种入的完整历史三账本；换轮和容器重启不重置次数。
    current_requests = ROOT / 'model-requests.jsonl'
    current_responses = ROOT / 'model-responses.jsonl'
    current_ledger = ROOT / 'attempt-ledger.jsonl'
    receipts = {}
    new_costs = {}
    known_cost_total = Decimal('0')
    reserved_inflight = set()
    if current_responses.exists():
        for line in current_responses.read_text().splitlines():
            if line.strip():
                saved = json.loads(line)
                prior = receipts.get(saved['call_id'])
                if prior is not None and prior != saved:
                    raise RuntimeError('GLOBAL_CURRENT_DUPLICATE_RECEIPT_MISMATCH')
                receipts[saved['call_id']] = saved
                response = saved.get('response')
                response_id = response.get('id') if isinstance(response, dict) else None
                if response_id and response_id not in baseline_response_ids:
                    cost = receipt_cost(response)
                    prior_cost = new_costs.get(response_id)
                    if prior_cost is not None and prior_cost != cost:
                        raise RuntimeError('GLOBAL_DUPLICATE_RESPONSE_COST_MISMATCH')
                    if prior_cost is None:
                        new_costs[response_id] = cost
                        known_cost_total += cost
    sent_hashes = {}
    if current_requests.exists():
        for line in current_requests.read_text().splitlines():
            if not line.strip():
                continue
            sent = json.loads(line)
            digest = request_hash(sent['request'])
            if digest != sent['request_sha256']:
                raise RuntimeError('GLOBAL_CURRENT_REQUEST_HASH_MISMATCH')
            call_id = sent['call_id']
            prior = seen_calls.get(call_id)
            if prior is not None and prior != (digest, sent['request']):
                raise RuntimeError('GLOBAL_CURRENT_CALL_ID_MISMATCH')
            saved = receipts.get(sent['call_id'])
            if saved and saved['request'] != sent['request']:
                raise RuntimeError('GLOBAL_CURRENT_RECEIPT_HASH_MISMATCH')
            if saved and saved.get('request_sha256', digest) != digest:
                raise RuntimeError('GLOBAL_CURRENT_RECEIPT_HASH_MISMATCH')
            response_id = saved.get('response', {}).get('id') if saved else None
            if prior is None and (digest, response_id) not in legacy_receipts:
                counts[digest] = counts.get(digest, 0) + 1
            seen_calls[call_id] = (digest, sent['request'])
            sent_hashes[call_id] = digest
            attempt = sent.get('attempt', 1)
            if type(attempt) is not int or attempt < 1:
                raise RuntimeError('GLOBAL_CURRENT_ATTEMPT_MISMATCH')
            counts[digest] = max(counts.get(digest, 0), attempt)
            if saved and saved.get('response', {}).get('id'):
                cache.setdefault(digest, saved['response'])
                cache_sources[digest] = str(current_responses)
            elif not saved or saved.get('error', {}).get('status_code') not in (402, 429):
                blocked.add(digest)
    if current_ledger.exists():
        for line in current_ledger.read_text().splitlines():
            if not line.strip():
                continue
            saved = json.loads(line)
            digest = saved['request_sha256']
            call_id = saved['call_id']
            if sent_hashes.get(call_id) != digest:
                raise RuntimeError('GLOBAL_CURRENT_LEDGER_REQUEST_MISMATCH')
            attempt = saved.get('attempt')
            if type(attempt) is not int or attempt < 1:
                raise RuntimeError('GLOBAL_CURRENT_LEDGER_ATTEMPT_MISMATCH')
            counts[digest] = max(counts.get(digest, 0), attempt)
            if saved['state'] == 'outcome_unknown':
                blocked.add(digest)
            elif saved['state'] == 'recorded' and saved.get('response_id'):
                receipt = receipts.get(call_id)
                if not receipt or receipt.get('response', {}).get('id') != saved['response_id']:
                    raise RuntimeError('GLOBAL_CURRENT_LEDGER_RECEIPT_MISMATCH')
            elif saved['state'] not in {'dispatching', 'recorded'}:
                raise RuntimeError('GLOBAL_CURRENT_LEDGER_STATE_MISMATCH')
    publisher_original = SceneResultPublisher.project
    def recorded_publish(self, *args, **kwargs):
        try:
            return publisher_original(self, *args, **kwargs)
        except (KeyError, TypeError, AttributeError) as error:
            write_stop_loss({
                'reason': 'GLOBAL_NATIVE_PUBLISHER_INTEGRATION_ERROR',
                'error_type': type(error).__name__, 'error': str(error)},
                global_failure=True)
            raise
    SceneResultPublisher.project = recorded_publish

    request_original = TemporalVLMInspector._request
    inspect_original = TemporalVLMInspector.inspect
    complete_original = SceneRuntime.complete_inspection
    evidence_original = SceneState.observe_event

    async def recorded_request(self, payload, headers, *, timing):
        nonlocal known_cost_total
        require_audit_writable()
        if (ROOT / 'stop-loss.json').exists() or (ROOT / 'balance-blocked.json').exists():
            raise RuntimeError('GLOBAL_STOP_LOSS_BEFORE_DISPATCH')
        request = no_images(payload)
        digest = request_hash(request)
        with LOCK:
            require_audit_writable()
            if (ROOT / 'stop-loss.json').exists() or (ROOT / 'balance-blocked.json').exists():
                raise RuntimeError('GLOBAL_STOP_LOSS_BEFORE_DISPATCH')
            if digest in blocked or digest in in_flight:
                reason = 'unresolved_same_input_request'
                blocked_row = {'reason': 'question_request_blocked',
                               'blocked_reason': reason, 'request_sha256': digest,
                               'stream_session_id': timing.get('stream_session_id'),
                               'context': dict(timing), 'paid_dispatch': False}
                append('question-blocked.jsonl', blocked_row)
                write_stop_loss(blocked_row)
                raise RuntimeError('QUESTION_UNRESOLVED_SAME_INPUT_REQUEST')
            if digest in cache:
                append('cache-hits.jsonl', {'request_sha256': digest, 'context': dict(timing),
                       'response_id': cache[digest]['id'], 'source': cache_sources[digest], 'paid_dispatch': False})
                return copy.deepcopy(cache[digest])
            if counts.get(digest, 0) >= 3:
                blocked_row = {'reason': 'question_request_blocked',
                               'blocked_reason': 'same_input_attempts_exhausted',
                               'request_sha256': digest,
                               'stream_session_id': timing.get('stream_session_id'),
                               'context': dict(timing), 'paid_dispatch': False}
                append('question-blocked.jsonl', blocked_row)
                write_stop_loss(blocked_row)
                raise RuntimeError('QUESTION_SAME_INPUT_ATTEMPTS_EXHAUSTED')
            if (known_cost_total
                    + dispatch_reserve * (len(reserved_inflight) + 1)
                    > known_cost_limit):
                write_stop_loss({
                    'reason': 'known_model_cost_dispatch_limit',
                    'known_cost_usd': str(known_cost_total),
                    'reserved_inflight_usd': str(dispatch_reserve * len(reserved_inflight)),
                    'next_reservation_usd': str(dispatch_reserve),
                    'limit_usd': str(known_cost_limit),
                })
                raise RuntimeError('GLOBAL_KNOWN_MODEL_COST_LIMIT')
            counts[digest] = counts.get(digest, 0) + 1
            in_flight.add(digest)
            reserved_inflight.add(digest)
        started = time.monotonic()
        row = {"started_epoch": time.time(), "request": request, 'request_sha256': digest,
               'call_id': str(uuid4()), 'attempt': counts[digest],
               "context": dict(timing)}
        append("model-requests.jsonl", row)
        append('attempt-ledger.jsonl', {'call_id': row['call_id'], 'request_sha256': digest,
               'attempt': row['attempt'], 'state': 'dispatching', 'context': dict(timing)})
        try:
            response = await request_original(self, payload, headers, timing=timing)
            row["response"] = response
            return response
        except BaseException as error:
            row["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
                            "status_code": getattr(error, "status_code", None)}
            if getattr(error, "status_code", None) == 402 or getattr(error, 'code', None) == 402:
                write_json_file(ROOT / "balance-blocked.json", row["error"])
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            row["timing"] = dict(timing)
            append("model-responses.jsonl", row)
            known_rejection = row.get('error', {}).get('status_code') in (402, 429)
            append('attempt-ledger.jsonl', {'call_id': row['call_id'], 'request_sha256': digest,
                   'attempt': row['attempt'], 'response_id': row.get('response', {}).get('id'),
                   'state': 'recorded' if row.get('response', {}).get('id') or known_rejection else 'outcome_unknown'})
            with LOCK:
                in_flight.discard(digest)
                reserved_inflight.discard(digest)
                if row.get('response', {}).get('id'):
                    cache[digest] = copy.deepcopy(row['response'])
                    cache_sources[digest] = str(current_responses)
                    response = row['response']
                    response_id = response['id']
                    if response_id not in baseline_response_ids:
                        try:
                            cost = receipt_cost(response)
                        except RuntimeError as error:
                            write_stop_loss({
                                'reason': str(error), 'response_id': response_id,
                            }, global_failure=True)
                        else:
                            prior_cost = new_costs.get(response_id)
                            if prior_cost is not None and prior_cost != cost:
                                write_stop_loss({
                                    'reason': 'GLOBAL_DUPLICATE_RESPONSE_COST_MISMATCH',
                                    'response_id': response_id,
                                }, global_failure=True)
                            elif prior_cost is None:
                                new_costs[response_id] = cost
                                known_cost_total += cost
                                if cost > dispatch_reserve or known_cost_total >= known_cost_limit:
                                    write_stop_loss({
                                        'reason': 'model_cost_reservation_exceeded'
                                        if cost > dispatch_reserve else 'known_model_cost_limit',
                                        'response_id': response_id,
                                        'response_cost_usd': str(cost),
                                        'known_cost_usd': str(known_cost_total),
                                        'limit_usd': str(known_cost_limit),
                                    }, global_failure=True)
                elif not known_rejection:
                    blocked.add(digest)

    def recorded_inspect(self, job):
        global ERRORS
        require_audit_writable()
        row = {"stream_session_id": job.stream_session_id,
               "request_id": job.request.request_id,
               "start_seconds": job.request.start_seconds,
               "end_seconds": job.request.end_seconds,
               "requested_event_labels": list(job.request.event_labels),
               "requested_subject_ids": list(job.request.subject_ids),
               "allowed_subject_ids": sorted({sid for _, sid, _ in job.subject_boxes}),
               "source_frames": [{"frame_sequence": frame.frame_sequence,
                                  "timestamp_seconds": frame.timestamp_seconds}
                                 for frame in job.frames],
               "evidence_after_frame_sequence": job.evidence_after_frame_sequence,
               "started_epoch": time.time()}
        append("native-window-starts.jsonl", row)
        try:
            result = inspect_original(self, job)
            row.update(status="completed", verdicts=[asdict(verdict) for verdict in result])
            with LOCK:
                ERRORS = 0
            return result
        except Exception as error:
            row.update(status="failed", error={"type": type(error).__name__,
                       "code": getattr(error, "code", None),
                       "status_code": getattr(error, "status_code", None),
                       "validation_reason": getattr(error, "validation_reason", None)})
            with LOCK:
                ERRORS += 1
                if ERRORS >= 3:
                    write_stop_loss({
                        "reason": "three_consecutive_observation_errors", **row})
            raise
        finally:
            row["finished_epoch"] = time.time()
            append("native-windows.jsonl", row)

    def recorded_evidence(self, event_label, subject_ids, **kwargs):
        accepted = evidence_original(self, event_label, subject_ids, **kwargs)
        append("native-evidence.jsonl", {
            "stream_session_id": self.stream_session_id,
            "event_label": event_label, "subject_ids": list(subject_ids),
            "accepted": bool(accepted), **kwargs,
        })
        return accepted

    def recorded_complete(self, request_id, verdicts, *, success=True, completion_counts=None):
        with self._lock:
            job = self._jobs.get(request_id)
        counts = {} if completion_counts is None else completion_counts
        completed = complete_original(self, request_id, verdicts, success=success,
                                      completion_counts=counts)
        append("native-completions.jsonl", {
            "stream_session_id": self.stream_session_id,
            "request_id": request_id, "success": success,
            "completed": bool(completed), "counts": dict(counts),
            "start_seconds": job.request.start_seconds if job else None,
            "end_seconds": job.request.end_seconds if job else None,
            "scene": self.scene.snapshot(include_history=True),
        })
        return completed

    TemporalVLMInspector._request = recorded_request
    TemporalVLMInspector.inspect = recorded_inspect
    SceneState.observe_event = recorded_evidence
    SceneRuntime.complete_inspection = recorded_complete
    INSTALLED = True
