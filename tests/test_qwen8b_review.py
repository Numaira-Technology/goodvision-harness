"""Frozen Qwen contract regressions using synthetic inputs and intercepted HTTP."""
import io
import json

import pytest

pytest.importorskip("fcntl", reason="Qwen evaluation is a Linux/macOS workflow")

from tools.qwen8b import eval_local, prepare_vllm_bundle
from tools.unified_benchmark import workflow
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import digest, file_hash, read_receipt
from tools.unified_benchmark.retry import Ledger, Queue, RequestFailure


from helpers.qwen_bundle import frozen_host_bundle


@pytest.fixture
def host(tmp_path):
    bundle = tmp_path / "bundle"
    config, unit = frozen_host_bundle(bundle)
    ledger = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True})
    client = eval_local.LocalTransport(bundle, config, "http://127.0.0.1:9", tmp_path / "run", ledger)
    yield bundle, config, unit, ledger, client
    ledger.close()


def good_response(model):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":true}'}}]}


@pytest.mark.parametrize("model", [None, "", 42, "other/model", False])
def test_local_model_identity_is_never_invented(host, monkeypatch, model):
    _, config, unit, ledger, client = host
    payload = good_response(config['model'])
    if model is None:
        payload.pop('model')
    else:
        payload['model'] = model
    monkeypatch.setattr(client, 'call', lambda body: payload)
    ledger.append({'type': 'intent', 'unit': unit['id'], 'attempt': 1, 'request_sha256': unit['request_sha256']})
    with pytest.raises(RequestFailure) as error:
        client(unit)
    assert error.value.fatal is True
    metadata = error.value.metadata
    assert metadata.get('actual_model') != config['model']
    assert file_hash(client.output / metadata['receipt']) == metadata['receipt_sha256']


@pytest.mark.parametrize('bad_body', [b'[]', b'null', b'{"model":"qwen/qwen3-vl-8b-instruct","x":NaN}',
                                     b'<html>bad upstream</html>', b'\xff', b'x' * (2 * 1024 * 1024 + 1),
                                     'incomplete_http', 'bad_status'])
def test_local_bad_responses_retry_three_times_without_stopping_good_units(host, monkeypatch, bad_body):
    bundle, config, unit, ledger, client = host
    good = unit | {'id': 'good', 'question': 'GOOD synthetic question'}
    good['request_sha256'] = digest(request_body(config, good, bundle)[1])
    class Response(io.BytesIO):
        status = 200
    class Truncated(Response):
        def read(self, size=-1):
            raise eval_local.http.client.IncompleteRead(b'{"model":', 10)
    calls = []
    def urlopen(request, **kwargs):
        is_good = b'GOOD synthetic question' in request.data
        calls.append(is_good)
        if not is_good and bad_body == 'incomplete_http':
            return Truncated(b'')
        if not is_good and bad_body == 'bad_status':
            raise eval_local.http.client.BadStatusLine('synthetic invalid status')
        return Response(json.dumps(good_response(config['model'])).encode() if is_good else bad_body)
    monkeypatch.setattr(eval_local.urllib.request, 'urlopen', urlopen)
    now = [0.0]
    result = Queue(ledger, 'choice', client, slots=1, clock=lambda: now[0],
                   sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)).run([unit, good])
    assert result[unit['id']] == {} and result['good']['fight']['answer'] == 'yes'
    assert calls.count(False) == 3 and calls.count(True) == 1
    assert len([r for r in ledger.rows if r['type'] == 'result']) == 4
    for row in (r for r in ledger.rows if r['type'] == 'result'):
        metadata = row['metadata']
        assert file_hash(client.output / metadata['receipt']) == metadata['receipt_sha256']
        read_receipt(client.output / metadata['receipt'])


@pytest.mark.parametrize('name', ['tools/qwen8b/eval_local.py', 'tools/qwen8b/serve.py',
    'tools/qwen8b/serve_vllm.py', 'tools/models.yaml', 'tools/env_source.py',
    'tools/start_qwen8b_r2.py', 'tools/start_qwen8b_vllm.py'])
def test_qwen_execution_sources_are_bound_before_local_inference(host, tmp_path, monkeypatch, name):
    bundle, _, _, _, _ = host
    active = workflow.source_files()
    changed = tmp_path / 'changed-source'
    changed.write_bytes(active[name].read_bytes() + b'\n# synthetic source mismatch\n')
    monkeypatch.setattr(workflow, 'source_files', lambda: active | {name: changed})
    def forbidden(*args, **kwargs):
        pytest.fail('source mismatch must fail before inference')
    monkeypatch.setattr(eval_local, 'execute', forbidden)
    with pytest.raises(ValueError, match='active source differs'):
        eval_local.main(['--bundle', str(bundle), '--output', str(tmp_path / 'untouched'),
            '--endpoint', 'http://127.0.0.1:9', '--authorization-note', 'synthetic', '--execute'])
    assert not (tmp_path / 'untouched').exists()


def test_vllm_derivation_verifies_source_before_copy(host, tmp_path):
    bundle, _, _, _, _ = host
    (bundle / 'frame.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bundle file differs'):
        prepare_vllm_bundle.prepare(bundle, tmp_path / 'copy')
    assert not (tmp_path / 'copy').exists()


def test_local_continuous_profile_cannot_bypass_calibration(host, tmp_path):
    bundle, config, _, _, _ = host
    path = bundle / 'configs/qwen8b_r2_host.json'
    path.write_text(json.dumps(config | {'kind': 'probability'}))
    with pytest.raises(ValueError, match='hard-answer'):
        eval_local.execute('qwen8b_r2_host', bundle, tmp_path, 'http://127.0.0.1:9', 'synthetic', 'test')
    assert not (tmp_path / 'qwen8b_r2_host').exists()
