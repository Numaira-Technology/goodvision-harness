"""正式双检测器评测的共享控制；不包含探针、模型试调用或独立测试。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.name
WORKSPACE = ROOT.parents[1]
NETWORK = 'f35225df3a67e006a294b37ac6eb53605424f9aabe0419123096c9517909e8ef'
DOCKER = '/usr/local/bin/docker'
IMAGE = 'sha256:3b9a41535c81fe6290d2282ff2a252861432fc500e7119794d5fd4008cbb06a3'
GPU_IMAGE = 'sha256:7ff2a7411640f77121d7dd755c9b0445d56028842ad33009c3dbca80a48fdbee'
OUTPUT = ROOT / 'output'
BASE = 'http://127.0.0.1:18525'
OPEN = build_opener(ProxyHandler({}))
ENV = dict(os.environ)

def load(path):
    return json.loads(Path(path).read_text())

def rows(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)

def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

def update(**fields):
    state = load(ROOT / 'run.json') if (ROOT / 'run.json').exists() else {}
    state.update(fields, updated_epoch=time.time())
    save(ROOT / 'run.json', state)
    print(json.dumps(fields, ensure_ascii=False), flush=True)

def command(args, timeout=120, check=True):
    result = subprocess.run(args, env=ENV, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'COMMAND_FAILED {args[0]} {args[1:2]}: {result.stderr[-1800:]}')
    return result

def docker(*args, **kwargs):
    return command([DOCKER, *map(str, args)], **kwargs)

def owned(name):
    result = docker('inspect', name, '--format', '{{json .}}', check=False)
    if result.returncode:
        if 'no such' in result.stderr.lower():
            return None
        raise RuntimeError('GLOBAL_RESOURCE_READ_FAILURE: ' + name)
    info = json.loads(result.stdout)
    if info.get('Config', {}).get('Labels', {}).get('experiment') != PREFIX:
        raise RuntimeError('GLOBAL_RESOURCE_OWNERSHIP_MISMATCH: ' + name)
    return info

def logs(name):
    result = docker('logs', name, check=False)
    return result.stdout + result.stderr

def local_question_stop(sid):
    path = OUTPUT / 'stop-loss.json'
    if not path.exists() or not sid:
        return None
    value = load(path)
    if value.get('stream_session_id') != sid:
        return None
    if value.get('reason') == 'question_request_blocked':
        allowed = {'unresolved_same_input_request', 'same_input_attempts_exhausted'}
        matches = [row for row in rows(OUTPUT / 'question-blocked.jsonl')
                   if row == value and row.get('blocked_reason') in allowed and row.get('paid_dispatch') is False]
        return value if matches else None
    if value.get('reason') != 'three_consecutive_observation_errors':
        return None
    error = value.get('error', {})
    known = (error.get('type') == 'ModelAPIError' and error.get('code') == 'invalid_response') or error.get('status_code') in (400, 429)
    unknown = unknown_for_session(sid)
    return value if known or unknown else None

def unknown_for_session(sid):
    terminal = {r['call_id']: r['state'] for r in rows(OUTPUT / 'attempt-ledger.jsonl')}
    return [r['call_id'] for r in rows(OUTPUT / 'model-requests.jsonl')
            if r.get('context', {}).get('stream_session_id') == sid and terminal.get(r['call_id']) == 'outcome_unknown']

def stop_loss(sid=None):
    for directory in (ROOT, OUTPUT, ROOT / 'gpu/output'):
        for name in ('cancel-requested.json', 'balance-blocked.json', 'audit-io-failed.json', 'stop-loss.json'):
            path = directory / name
            if path.exists():
                if path == OUTPUT / 'stop-loss.json' and local_question_stop(sid):
                    raise RuntimeError('QUESTION_KNOWN_CONTRACT_OR_UNKNOWN_FAILURE')
                raise RuntimeError('GLOBAL_STOP_LOSS: ' + str(path))
    launch = load(ROOT / 'launch-intent.json')
    if time.time() >= launch['deadline_epoch']:
        raise RuntimeError('GLOBAL_AUTHORIZED_24_HOUR_DEADLINE')
    detector = owned(PREFIX + '-detector')
    if detector and not detector['State']['Running']:
        raise RuntimeError('GLOBAL_SHARED_DETECTOR_EXITED')
    if sid and unknown_for_session(sid):
        raise RuntimeError('QUESTION_UNKNOWN_REQUEST_BLOCKED')

def api(path, data=None, timeout=30, sid=None):
    stop_loss(sid)
    request = Request(BASE + path, data=None if data is None else json.dumps(data).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        with OPEN.open(request, timeout=timeout) as response:
            result = json.load(response)
    except OSError as error:
        raise RuntimeError('QUESTION_CONTROL_NETWORK_FAILURE: ' + str(error)) from error
    stop_loss(sid)
    return result

def check_frozen():
    for relative, expected in load(ROOT / 'source-manifest.json')['sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('GLOBAL_FROZEN_SOURCE_MISMATCH: ' + relative)
