"""Synthetic staged-run contracts; no real model results or network calls."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.clip_benchmark import controls as c, run, scoring
from tools.clip_benchmark.protocol import PROMPTS, REVISIONS

MODEL = next(iter(REVISIONS))


def save(path, data):
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def config(tmp_path):
    sidecar = {"schema_version": 1, "status": "validation_membership_and_labels_frozen_with_recorded_limits",
               "train": {"status": "unknown"}, "development": {"status": "unknown"}}
    config = {"interval_seconds": 1.0, "slots": 12}
    for part, repeats in (("validation", 6), ("test", 2)):
        media = tmp_path / part
        media.mkdir()
        samples, files, members = [], [], []
        for event in scoring.BASE_EVENTS:
            for i in range(repeats):
                name = f'{part}-{event}-{i}.mp4'
                (media / name).write_bytes(name.encode())
                row = {"number": len(samples) + 1, "filename": name, "event": event,
                       "question": f"Is {event} happening?", "proposed_answer": "yes" if i % 2 else "no"}
                samples.append(row)
                checksum = c.digest(media / name)
                files.append(row | {"sha256": checksum})
                members.append({"filename": name, "sha256": checksum,
                                "source_video_group": name, "question_numbers": [row["number"]]})
        samples_path = Path(save(tmp_path / f'{part}-samples.json', {"samples": samples}))
        manifest_path = Path(save(tmp_path / f'{part}-manifest.json',
                             {"status": "human_reviewed", "samples_json_sha256": c.digest(samples_path), "files": files}))
        config[part] = {"samples": str(samples_path), "manifest": str(manifest_path), "media": [str(media)]}
        sidecar[part] = {"questions": len(samples), "videos": len(files), "members": members,
                         "samples_json_sha256": c.digest(samples_path), "manifest_sha256": c.digest(manifest_path)}
    config["split_manifest"] = save(tmp_path / 'split.json', sidecar)
    config["reference_thresholds"] = save(tmp_path / 'old.json',
                                         {model: dict.fromkeys(PROMPTS, 0.1) for model in REVISIONS})
    return Path(save(tmp_path / 'config.json', config))


def planned(media, decode=False):
    return {row["filename"]: {"planned_1fps_observations": 1, "source_fps": 30.0, "frame_count": 1}
            for row in media}


def fake_runtime(monkeypatch):
    closed = []

    class Encoder:
        def __init__(self, model, cache):
            self.metadata = {"model": model, "precision": "float32", "load_and_text_seconds": 0.01}

        def close(self):
            closed.append(True)

    def streams(media, encoder, thresholds, output, slots, interval):
        assert slots == 12 and interval == 1.0
        assert all(set(row) == {"filename", "video_path"} for row in media)
        rows, states = [], []
        for item in media:
            positive = int(item["filename"].rsplit('-', 1)[1].split('.')[0]) % 2
            row = {"filename": item["filename"], "quality_reference": True, "errors": None,
                   "timestamp_seconds": 0.0, "frame_index": 0,
                   "predictions": {event: {"margin": 0.2 if positive else -0.2} for event in PROMPTS}}
            rows.append(row)
            states.append({"filename": item["filename"], "quality_reference": True,
                           "execution_complete": True, "error": None, "completed": 1,
                           "opportunities": 1, "busy_missed": 0, "late_missed": 0, "local_failures": 0})
        (output / 'clip-records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        summary = {"execution_complete": True, "shared_failure": None, "streams": states}
        save(output / 'run.json', encoder.metadata)
        save(output / 'completion.json', summary)
        return summary

    monkeypatch.setattr(c, 'gpu_environment', lambda: {"gpu": "synthetic only"})
    monkeypatch.setattr(c, 'inspect_media', planned)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, 'tools.clip_benchmark.runtime', SimpleNamespace(ClipEncoder=Encoder, run_streams=streams))
    return closed


def validation_run(config, tmp_path, monkeypatch):
    closed = fake_runtime(monkeypatch)
    context = c.load_inputs(config)
    output = tmp_path / 'val-run'
    assert run.collect(context, MODEL, 'validation', output, tmp_path / 'cache') == 0
    assert closed == [True]
    return context, output


def test_staged_validation_freeze_test_and_three_local_reports(config, tmp_path, monkeypatch):
    context, val = validation_run(config, tmp_path, monkeypatch)
    selected = tmp_path / 'selection'
    assert run.select(context, MODEL, val / 'bundle.json', selected) == 0
    path = selected / 'frozen-thresholds.json'
    artifact = c.verify_threshold_artifact(path, c.digest(path), context['contract'], MODEL)
    assert artifact['thresholds'] == dict.fromkeys(PROMPTS, 0.0)
    assert artifact['historical_thresholds'] == dict.fromkeys(PROMPTS, 0.1)
    output = tmp_path / 'test-run'
    assert run.collect(context, MODEL, 'test', output, tmp_path / 'cache', artifact) == 0
    assert len(list((output / 'reports').glob('*/base4/summary.json'))) == 3
    assert json.loads((output / 'bundle.json').read_text())['split'] == 'test'


def test_preflight_offline_and_no_overwrite(config, tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'inspect_media', planned)
    monkeypatch.setattr(c, 'gpu_environment', lambda: pytest.fail('offline must not touch CUDA'))
    output = tmp_path / 'preflight.json'
    args = ['preflight', '--config', str(config), '--output', str(output)]
    assert run.main(args) == 0
    report = json.loads(output.read_text())
    assert report['inference_calls'] == report['model_downloads'] == 0
    with pytest.raises(FileExistsError):
        run.main(args)


@pytest.mark.parametrize('mutation', ['media', 'source', 'cadence', 'label'])
def test_frozen_input_guards(config, mutation):
    data = json.loads(config.read_text())
    if mutation == 'media':
        next(Path(data['validation']['media'][0]).glob('*.mp4')).write_bytes(b'changed')
    elif mutation == 'source':
        path = Path(data['split_manifest'])
        split = json.loads(path.read_text())
        split['test']['members'][0]['source_video_group'] = split['validation']['members'][0]['source_video_group']
        save(path, split)
    elif mutation == 'cadence':
        data['interval_seconds'] = 0.8
        save(config, data)
    else:
        path = Path(data['validation']['samples'])
        samples = json.loads(path.read_text())
        samples['samples'][0]['proposed_answer'] = 'yes'
        save(path, samples)
    with pytest.raises(ValueError):
        c.load_inputs(config)


@pytest.mark.parametrize('mutation', ['truncated', 'duplicate', 'busy', 'nonfinite', 'frame_identity'])
def test_partial_validation_cannot_select(config, tmp_path, monkeypatch, mutation):
    context, output = validation_run(config, tmp_path, monkeypatch)
    completion = json.loads((output / 'completion.json').read_text())
    records_path = output / 'clip-records.jsonl'
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    if mutation == 'truncated':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(rows[0])
    elif mutation == 'busy':
        completion['streams'][0]['busy_missed'] = 1
    elif mutation == 'nonfinite':
        rows[0]['predictions']['fight']['margin'] = float('nan')
    else:
        rows[0]['frame_index'] = 9
    records_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        c.validation_complete(records_path, completion, context['parts']['validation']['samples'],
                              planned(context['parts']['validation']['media']))


def test_stale_scores_or_changed_contract_cannot_select(config, tmp_path, monkeypatch):
    context, output = validation_run(config, tmp_path, monkeypatch)
    other = copy.deepcopy(context['contract'])
    other['protocol']['sampling_fps'] = 1.25
    with pytest.raises(ValueError, match='contract differs'):
        c.verify_bundle(output / 'bundle.json', other, MODEL, 'validation')
    with (output / 'clip-records.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        c.verify_bundle(output / 'bundle.json', context['contract'], MODEL, 'validation')


def test_test_requires_original_threshold_hash_and_hardware(config, tmp_path, monkeypatch):
    context, val = validation_run(config, tmp_path, monkeypatch)
    selected = tmp_path / 'selection'
    run.select(context, MODEL, val / 'bundle.json', selected)
    path = selected / 'frozen-thresholds.json'
    with pytest.raises(ValueError, match='checksum'):
        c.verify_threshold_artifact(path, '0' * 64, context['contract'], MODEL)
    artifact = c.verify_threshold_artifact(path, c.digest(path), context['contract'], MODEL)
    artifact['hardware'] = {'gpu': 'different'}
    with pytest.raises(ValueError, match='environment differs'):
        run.collect(context, MODEL, 'test', tmp_path / 'test-output', tmp_path / 'cache', artifact)
    assert not (tmp_path / 'test-output').exists()


def test_unbounded_optimum_stops_and_strict_comparator_midpoint():
    samples = [{'filename': str(i), 'event': 'fight', 'proposed_answer': 'yes' if i % 2 else 'no'} for i in range(6)]
    indexed = {str(i): [{'errors': None, 'predictions': {'fight': {'margin': 0.2 if i % 2 else -0.2}}}] for i in range(6)}
    threshold, interval, _, _ = scoring._optimal_midpoint(samples, indexed, MODEL)
    assert threshold == 0.0 and interval['lower'] == -0.2 and interval['upper'] == 0.2
    for rows in indexed.values():
        rows[0]['predictions']['fight']['margin'] = 0.2
    with pytest.raises(ValueError, match='unbounded'):
        scoring._optimal_midpoint(samples, indexed, MODEL)
