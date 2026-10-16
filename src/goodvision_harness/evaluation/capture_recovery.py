"""Import exact terminal evidence without model calls or hardware-identity changes."""
import copy
import hashlib
import json
from pathlib import Path
import shutil


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_config(config):
    value = copy.deepcopy(config)
    value.pop('stream_session_id', None)
    value.pop('model_ledger', None)
    # Runtime, source bytes, model settings, and window budgets must still match.
    return value


def same_semantics(previous, current):
    if semantic_config(previous) != semantic_config(current):
        raise RuntimeError('reused source semantic configuration changed')


def source_files(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError('reused source must be an existing non-symlink directory')
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('reused source symlink forbidden')
        if path.is_file():
            files[str(path.relative_to(directory))] = digest(path)
    return files


def check_terminal(directory, terminal, previous, owner, recovery, config_sha256):
    directory = Path(directory)
    if (terminal['video_id'] != previous['video_id']
            or terminal['stream_session_id'] != previous['stream_session_id']
            or terminal['status'] not in {'captured', 'skipped', 'incomplete'}):
        raise RuntimeError('reused source terminal identity mismatch')
    manifest = load(directory / 'manifest.json')
    if (manifest['video_id'] != previous['video_id']
            or manifest['stream_session_id'] != previous['stream_session_id']
            or manifest['provenance']['source_manifest_sha256'] != previous['source_manifest_sha256']
            or manifest['provenance'].get('run_config_sha256') != config_sha256
            or manifest['provenance'].get('frozen_runtime_sha256') != previous['provenance']['frozen_runtime_sha256']
            or (terminal.get('resource_instance_id') is not None
                and terminal['resource_instance_id'] != owner['instance_id'])):
        raise RuntimeError('reused source manifest mismatch')
    if terminal['status'] == 'captured':
        evidence = load(directory / 'capture.json')
        if (evidence['video_id'] != previous['video_id']
                or evidence['stream_session_id'] != previous['stream_session_id']
                or evidence.get('request_failure') is not False
                or evidence.get('failures')
                or not all(evidence.get(key) is True for key in
                    ('source_eof', 'tracking_drained', 'events_drained', 'fusion_drained'))
                or evidence['observation_manifest'].get('complete') is not True):
            raise RuntimeError('reused capture incomplete')
    elif terminal['status'] == 'incomplete':
        evidence = load(directory / 'capture.json')
        outcome = load(directory / 'outcome.json')
        cleanup = load(directory / 'cleanup.json')
        if (evidence['video_id'] != previous['video_id']
                or evidence['stream_session_id'] != previous['stream_session_id']
                or evidence.get('request_failure') is not True
                or not evidence.get('failures')
                or evidence['observation_manifest'].get('complete') is not False
                or not all(evidence.get(key) is True for key in
                    ('source_eof', 'tracking_drained', 'events_drained', 'fusion_drained'))
                or outcome.get('status') != 'incomplete'
                or outcome.get('video_id') != previous['video_id']
                or outcome.get('stream_session_id') != previous['stream_session_id']
                or outcome.get('failures') != evidence['failures']
                or outcome.get('final_answer_generated') is not False
                or cleanup.get('stream_session_id') != previous['stream_session_id']
                or cleanup.get('local_cleanup_confirmed') is not True
                or cleanup.get('remote_cleanup_confirmed') is not True
                or cleanup.get('requires_worker_reset') is not False
                or cleanup.get('errors')):
            raise RuntimeError('reused incomplete terminal lacks preserved failure or cleanup evidence')
    else:
        if ((directory / 'capture.json').exists()
                or (terminal.get('failure_policy') or {}).get('scope') != 'source'
                or recovery.get('safe_to_continue') is not True
                or recovery.get('instance_id') != owner['instance_id']
                or recovery.get('experiment_label') != owner['experiment_label']):
            raise RuntimeError('reused skip lacks confirmed source cleanup')


def prepare_reuse(*, source_directory, config_path, config_sha256,
                  terminal_path, terminal_sha256, owner_path, owner_sha256,
                  origin_round, recovery_path=None, recovery_sha256=None):
    """Build a plan entry from explicit, hash-pinned prior artifacts; no writes."""
    from .capture_source import _read_frozen
    previous = _read_frozen(config_path, config_sha256)
    terminal = _read_frozen(terminal_path, terminal_sha256)
    owner = _read_frozen(owner_path, owner_sha256)
    recovery = (_read_frozen(recovery_path, recovery_sha256)
                if recovery_path is not None else {})
    if owner.get("source_manifest_sha256") != previous["provenance"]["frozen_runtime_sha256"]:
        raise RuntimeError("reused original runtime ownership mismatch")
    check_terminal(source_directory, terminal, previous, owner, recovery, config_sha256)
    return {"source_directory": str(Path(source_directory).resolve()),
            "files_sha256": source_files(source_directory), "previous_config": previous,
            "terminal": terminal, "owner": owner, "recovery": recovery,
            "original_config_path": str(Path(config_path).resolve()), "original_config_sha256": config_sha256,
            "origin_round": str(origin_round), "origin_terminal_path": str(terminal_path),
            "origin_terminal_sha256": terminal_sha256, "origin_resource_sha256": owner_sha256,
            "original_instance_id": owner["instance_id"]}


def import_reused_source(reuse, current, destination):
    from .capture_source import _read_frozen
    original = _read_frozen(reuse['original_config_path'], reuse['original_config_sha256'])
    if original != reuse['previous_config']:
        raise RuntimeError('reused original configuration changed')
    same_semantics(reuse['previous_config'], current)
    if (reuse['owner'].get('source_manifest_sha256') !=
            reuse['previous_config']['provenance']['frozen_runtime_sha256']
            or reuse['owner']['instance_id'] != reuse['original_instance_id']):
        raise RuntimeError('reused runtime or hardware identity mismatch')
    source = Path(reuse['source_directory'])
    if source_files(source) != reuse['files_sha256']:
        raise RuntimeError('frozen reuse file hash mismatch')
    check_terminal(source, reuse['terminal'], reuse['previous_config'], reuse['owner'], reuse['recovery'],
                   reuse['original_config_sha256'])
    shutil.copytree(source, destination)
    if source_files(destination) != reuse['files_sha256']:
        raise RuntimeError('copied reuse file hash mismatch')
    return {**reuse['terminal'], 'capture_directory': str(destination), 'reused_capture': True,
            'resource_instance_id': reuse['original_instance_id'],
            'origin_round': reuse['origin_round'], 'origin_terminal_path': reuse['origin_terminal_path'],
            'origin_terminal_sha256': reuse['origin_terminal_sha256'],
            'new_inference_performed': False}
