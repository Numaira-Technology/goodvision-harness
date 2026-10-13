"""唯一正式冻结：源文件、同题队列、历史请求账本及成功缓存。"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import control as c

PRIOR = c.ROOT.parent / 'main-event-supplement30-ashley1-resume10-20261009'

def main():
    if (c.ROOT / 'preparation.json').exists() or (c.ROOT / 'launch-intent.json').exists():
        raise RuntimeError('准备入口已执行，禁止重跑')
    queue = c.load(c.ROOT / 'cohort/queue.json')['questions']
    if len(queue) != 182 or len({r['qa_id'] for r in queue}) != 182 or sum(r['geometry_only'] for r in queue) != 90:
        raise RuntimeError('GLOBAL_COHORT_MISMATCH')
    request_rows = c.rows(PRIOR / 'ashley/output/model-requests.jsonl')
    responses = c.rows(PRIOR / 'ashley/output/model-responses.jsonl')
    ledger = c.rows(PRIOR / 'ashley/output/attempt-ledger.jsonl')
    requests = {r['call_id']: r for r in request_rows}
    receipts = {r['call_id']: r for r in responses}
    terminal = {r['call_id']: r for r in ledger if r['state'] != 'dispatching'}
    dispatched = {r['call_id']: r for r in ledger if r['state'] == 'dispatching'}
    if set(requests) != set(receipts) or set(requests) != set(terminal) or set(requests) != set(dispatched):
        raise RuntimeError('GLOBAL_PRIOR_LEDGER_NOT_CLOSED')
    unknown = []
    for call, request in requests.items():
        if request['request_sha256'] != receipts[call]['request_sha256'] or request['request_sha256'] != terminal[call]['request_sha256']:
            raise RuntimeError('GLOBAL_PRIOR_HASH_MISMATCH')
        if terminal[call]['state'] == 'outcome_unknown':
            unknown.append({'call_id': call, 'request_sha256': request['request_sha256']})
        elif terminal[call]['state'] != 'recorded':
            raise RuntimeError('GLOBAL_PRIOR_TERMINAL_MISMATCH')
    if len(requests) != 14169 or len(unknown) != 3:
        raise RuntimeError('GLOBAL_PRIOR_EXPECTED_COUNTS_MISMATCH')
    shutil.copyfile(c.ROOT.parent / 'ashley-core116-native-resume21-20261004/media-metadata.jsonl',
                    c.ROOT / 'cohort/core116-media-metadata.jsonl')
    shutil.copyfile(PRIOR / 'media-metadata.json', c.ROOT / 'cohort/supplement-media-metadata.json')
    c.OUTPUT.mkdir(exist_ok=True)
    for name in ('model-requests.jsonl', 'model-responses.jsonl', 'attempt-ledger.jsonl'):
        target = c.OUTPUT / name
        if target.exists():
            raise RuntimeError('GLOBAL_LEDGER_TARGET_EXISTS')
        shutil.copyfile(PRIOR / 'ashley/output' / name, target)
    shutil.copyfile(PRIOR / 'ashley/previous-model-responses.jsonl', c.ROOT / 'previous-model-responses.jsonl')
    history = c.rows(c.ROOT / 'previous-model-responses.jsonl') + responses
    ids = sorted({r['response']['id'] for r in history if r.get('response', {}).get('id')})
    c.save(c.ROOT / 'baseline-response-ids.json', ids)
    c.save(c.ROOT / 'prior-ledger-audit.json', {'requests': len(requests), 'receipts': len(receipts),
           'dispatch': len(dispatched), 'recorded': len(requests) - len(unknown), 'unknown': unknown,
           'baseline_response_ids': len(ids), 'new_paid_requests': 0, 'source': str(PRIOR)})
    hashes = {}
    for dirname in ('source', 'gpu/source', 'extraction', 'scoring', 'public', 'private', 'cohort'):
        for path in sorted((c.ROOT / dirname).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc'}:
                hashes[str(path.relative_to(c.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('control.py', 'driver.py', 'run_batch.py', 'start.py', 'prepare.py', 'native_entry.py',
                 'extract_result.py', 'analyze.py', 'cohort_prepare.py', 'baseline-response-ids.json', 'previous-model-responses.jsonl', 'recovery-policy.json'):
        hashes[name] = hashlib.sha256((c.ROOT / name).read_bytes()).hexdigest()
    c.save(c.ROOT / 'source-manifest.json', {'sha256': hashes, 'created_epoch': time.time(),
           'method': 'ashley_native_siyu_main_rules_detector_ablation',
           'native_base': 'fca08e118c013d053ff880153dc6008584805e64',
           'detector_upstream': 'f9c82f57ad74513f79f44a329717ad9af9aaa497',
           'event_definitions': 51, 'models': ['yolo26-s', 'rfdetr-l'], 'scorer': 'Benchmark PR63'})
    c.save(c.ROOT / 'ready.json', {'manifest_sha256': hashlib.sha256((c.ROOT / 'source-manifest.json').read_bytes()).hexdigest(),
           'questions_per_arm': 182, 'geometry_only_per_arm': 90, 'event_vlm_per_arm': 92,
           'pilot_per_arm': 20, 'independent_tests_run': False})
    c.save(c.ROOT / 'preparation.json', {'phase': 'prepared_not_started', 'prepared_epoch': time.time(),
           'questions_per_arm': 182, 'max_gpu_instances': 1, 'duration_hours': 24,
           'model_known_cost_limit_usd': 20, 'gpu_actual_cost_limit_usd': 10})
    print(json.dumps(c.load(c.ROOT / 'ready.json')))

if __name__ == '__main__':
    main()
