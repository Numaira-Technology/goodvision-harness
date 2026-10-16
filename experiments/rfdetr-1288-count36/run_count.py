"""Replay frozen count inputs once per session; retain raw responses and failures."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
import httpx

ROOT = Path('/run')
sys.path.insert(0, str(ROOT / 'extraction'))
from unified_answers import answer_question

def load(p):
    return json.loads(p.read_text())

def save(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)

def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def guard():
    if os.environ.get('DEADLINE_EPOCH') and time.time() >= float(os.environ['DEADLINE_EPOCH']):
        raise RuntimeError('GLOBAL_APPROVED_WEEKLY_DEADLINE')
    if (ROOT / 'state/cancel-requested.json').exists():
        raise RuntimeError('GLOBAL_CANCEL_REQUESTED')

def call(client, operation, model, sid, **fields):
    guard()
    response = client.post('http://detector-remote:8093/v1/count',
        json={'operation': operation, 'model': model, 'sid': sid, **fields},
        # The gateway still stops frame inference at 74 seconds. Allow its
        # separate process-cleanup phase to finish and return the error receipt.
        timeout=335 if operation == 'open' else 115)
    if response.is_error:
        # Retain server errors without resending a stateful frame.
        detail = response.text[:12000]
        raise RuntimeError('QUESTION_MODEL_HTTP_' + str(response.status_code) + ': ' + detail)
    value = response.json()
    if value.get('sid') != sid or value.get('model') != model:
        raise RuntimeError('GLOBAL_RESPONSE_IDENTITY_MISMATCH')
    guard()
    return value

def main():
    model = sys.argv[1]
    if model != 'rfdetr-l':
        raise ValueError('Only RF-DETR-L is approved for this round')
    questions = [json.loads(s) for s in (ROOT / 'public/questions.jsonl').read_text().splitlines() if s]
    expected = load(ROOT / 'input-manifest.json')['sha256']
    selected=load(ROOT/'selected-parameters.json')['selected']
    preserved=load(ROOT/'preserved-completions.json').get(model,{})
    prior_attempts=load(ROOT/'attempt-history.json').get(model,{})
    skipped=load(ROOT/'preserved-skips.json').get(model,{})
    parameters={k:selected[k] for k in ('imgsz','confidence','max_det')}
    with httpx.Client(limits=httpx.Limits(max_keepalive_connections=0), trust_env=False) as client:
        for question in questions:
            guard()
            qid = question['qa_id']
            directory = ROOT / 'results' / model / qid
            if qid in preserved:
                path=directory/'outcome.json'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=preserved[qid]['outcome_sha256']:
                    raise RuntimeError('GLOBAL_PRESERVED_ANSWER_CHANGED')
                prior=load(path)
                if (prior.get('status')!='answered' or prior.get('model')!=model or prior.get('qa_id')!=qid
                    or not prior.get('source_eof') or not prior.get('tracking_drained')):
                    raise RuntimeError('GLOBAL_INVALID_PRESERVED_ANSWER')
                print(json.dumps({'model':model,'qa_id':qid,'status':'preserved_complete_not_replayed'}),flush=True)
                continue
            if qid in skipped:
                path=directory/'outcome.json'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=skipped[qid]['outcome_sha256']:
                    raise RuntimeError('GLOBAL_PRESERVED_SKIP_CHANGED')
                print(json.dumps({'model':model,'qa_id':qid,'status':'preserved_failure_skipped'}),flush=True)
                continue
            directory.mkdir(parents=True, exist_ok=False)
            sid = str(uuid.uuid4())
            started = time.time()
            started_mono=time.monotonic()
            record = {'qa_id': qid, 'model': model, 'sid': sid, 'started_epoch': started}
            save(directory / 'session.json', record)
            try:
                manifest_path = ROOT / 'frames' / qid / 'manifest.json'
                rel = str(manifest_path.relative_to(ROOT))
                if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected[rel]:
                    raise RuntimeError('GLOBAL_INPUT_MANIFEST_MISMATCH')
                manifest = load(manifest_path)
                if not manifest['source_eof'] or manifest['qa_id'] != qid:
                    raise RuntimeError('GLOBAL_SOURCE_EOF_MISSING')
                opened = call(client, 'open', model, sid, detector_parameters=parameters)
                if 'RTX 4090' not in opened.get('provenance', {}).get('actual_gpu_name', ''):
                    raise RuntimeError('GLOBAL_REQUIRED_4090_HARDWARE_MISMATCH')
                if model == 'sam3.1' and opened.get('provenance', {}).get('sam_internal_patch') is not False:
                    raise RuntimeError('GLOBAL_SAM_INTERNAL_PATCH_NOT_ALLOWED')
                save(directory / 'open.json', opened)
                observed_frames = []
                latencies=[]; inference_started=time.monotonic()
                for item in manifest['frames']:
                    guard()
                    data = (manifest_path.parent / item['file']).read_bytes()
                    if hashlib.sha256(data).hexdigest() != item['jpeg_sha256']:
                        raise RuntimeError('GLOBAL_JPEG_HASH_MISMATCH')
                    append(directory / 'requests.jsonl', {'sid': sid, **item, 'state': 'dispatching', 'epoch': time.time()})
                    requested=time.monotonic()
                    result = call(client, 'frame', model, sid, frame_sequence=item['source_frame_index'],
                        timestamp_seconds=item['timestamp_seconds'], jpeg_sha256=item['jpeg_sha256'],
                        jpeg_base64=base64.b64encode(data).decode('ascii'))
                    if (result.get('jpeg_sha256') != item['jpeg_sha256'] or
                        result.get('frame_sequence') != item['source_frame_index'] or
                        result.get('timestamp_seconds') != item['timestamp_seconds'] or
                        not str(result.get('actual_inference_device', '')).startswith('cuda')):
                        raise RuntimeError('GLOBAL_FRAME_RESPONSE_MISMATCH')
                    if parameters is not None and result.get('audit',{}).get('parameters')!=parameters:
                        raise RuntimeError('GLOBAL_SELECTED_PARAMETERS_MISMATCH')
                    elapsed=time.monotonic()-requested; latencies.append(elapsed)
                    append(directory / 'responses.jsonl', dict(result,frame_e2e_seconds=elapsed))
                    observed = []
                    for obj in result['objects']:
                        if obj['observation_state'] == 'predicted':
                            continue
                        if (obj['observation_state'] != 'observed' or obj['label'] != 'person' or
                            obj['last_observed_frame_sequence'] != item['source_frame_index'] or
                            abs(obj['last_observed_timestamp_seconds'] - item['timestamp_seconds']) > 1e-6):
                            raise RuntimeError('GLOBAL_OBSERVATION_CONTRACT_MISMATCH')
                        observed.append({'id': obj['track_id'], 'bbox': obj['bbox'],
                            'observation_state':'observed','label':'person'})
                    if len({o['id'] for o in observed}) != len(observed):
                        raise RuntimeError('GLOBAL_DUPLICATE_TRACK_ID')
                    observed_frames.append({'t': item['timestamp_seconds'], 'frame': item['source_frame_index'], 'subjects': observed})
                    save(ROOT / 'state/progress.json', {'model': model, 'qa_id': qid, 'frames_done': len(observed_frames),
                         'frames_total': len(manifest['frames']), 'last_timestamp': item['timestamp_seconds'], 'epoch': time.time()})
                closed = call(client, 'close', model, sid)
                save(directory / 'close.json', closed)
                if not closed.get('tracking_drained') or len(observed_frames) != len(manifest['frames']):
                    raise RuntimeError('GLOBAL_TRACKING_NOT_DRAINED')
                capture = {'video_id': question['video_id'], 'fps': 2, 'duration_seconds': manifest['last_source_timestamp'],
                    'frames': observed_frames, 'events': [], 'source_eof': True, 'tracking_drained': True,
                    'width': manifest['frames'][0]['input_size'][0], 'height': manifest['frames'][0]['input_size'][1]}
                save(directory / 'capture.json', capture)
                answer = answer_question(question,capture)
                save(directory/'answer.json',answer)
                if answer['status']!='answered': raise RuntimeError('GLOBAL_UNSUPPORTED_COUNT_SEMANTICS')
                prediction=answer['final_answer']
                processing=time.monotonic()-inference_started
                save(directory / 'outcome.json', {**record, 'status': 'answered', 'prediction': prediction,
                    'finished_epoch': time.time(), 'source_eof': True, 'tracking_drained': True,
                    'frames': len(observed_frames), 'input_manifest_sha256': expected[rel],
                    'method': 'shared_input_person_route_unified_evidence_count_v1','parser_version':answer['parser_version'],
                    'implementation': 'rf1288-count36-v1',
                    'model_provenance': opened['provenance'],
                    'detector_parameters':parameters,'vlm_calls': 0,'face_enabled':False,
                    'processing_wall_seconds':processing,'preparation_and_open_seconds':inference_started-started_mono,
                    'frame_e2e_seconds':latencies,
                    'source_time_span_seconds':manifest['last_source_timestamp'],
                    'latency_scope':'person_route_backpressure_including_network_not_full_harness'})
                print(json.dumps({'model': model, 'qa_id': qid, 'prediction': prediction, 'status': 'answered'}), flush=True)
            except Exception as exc:
                reason = type(exc).__name__ + ': ' + str(exc)
                # Only a model OOM or server-side operation deadline is a question failure.
                # IO, transport, identity and extraction errors stop the shared run.
                unsafe = not (
                    isinstance(exc, RuntimeError)
                    and 'QUESTION_MODEL_HTTP_503:' in reason
                    and ('CUDA out of memory' in reason or 'MODEL_OPERATION_TIMEOUT' in reason)
                    and 'GLOBAL_' not in reason
                )
                cleanup = None
                try:
                    reset = client.post('http://detector-remote:8093/v1/control/reset',
                        json={'experiment_label':os.environ['DETECTOR_EXPERIMENT_LABEL'],
                              'expected_model':model, 'expected_sid':sid}, timeout=90)
                    reset.raise_for_status()
                    cleanup = reset.json()
                    save(directory / 'cleanup.json', cleanup)
                    if (cleanup.get('reset_confirmed') is not True
                            or cleanup.get('remaining_model_pid') is not None
                            or cleanup.get('experiment_label') != os.environ['DETECTOR_EXPERIMENT_LABEL']):
                        raise RuntimeError('GLOBAL_RESET_UNCONFIRMED')
                except Exception as reset_error:
                    unsafe = True
                    save(directory / 'cleanup-failure.json',
                         {'error':type(reset_error).__name__ + ': ' + str(reset_error)})
                save(directory / 'outcome.json', {**record, 'status': 'pending' if unsafe else 'skipped',
                    'prediction': None, 'reason': reason, 'failed_epoch': time.time(),
                    'source_eof': False, 'tracking_drained': False,
                    'cleanup':cleanup, 'failure_stage': 'formal_model_or_extraction', 'new_attempts_this_round': 1,
                    'historical_attempt_sids': prior_attempts.get(qid, []),
                    'next_action': 'integrity_recovery_required' if unsafe else 'continue_next_question_fresh_session'})
                if unsafe:
                    raise
                print(json.dumps({'model': model, 'qa_id': qid, 'status': 'skipped',
                                  'reason': reason, 'continuing': True}), flush=True)
                continue

if __name__ == '__main__':
    main()

