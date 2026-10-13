"""两组同题正式执行；先行20题计入182，几何题不连接VLM。"""
from pathlib import Path
import json
import re
import time
import control as c
from extract_result import extract_question, session_features

ACTIVE = None

def stop_container(name, directory):
    info = c.owned(name)
    if info and info['State']['Running']:
        c.docker('stop', '--time', '180', name, timeout=240)
    info = c.owned(name)
    if info:
        (directory / 'harness.log').write_text(c.logs(name))
        if info['State']['Running']:
            raise RuntimeError('GLOBAL_CONTAINER_NOT_STOPPED')
        if info['State'].get('ExitCode') == 74:
            raise RuntimeError('GLOBAL_AUDIT_IO_FAILURE')

def start_container(arm, geometry, stage, index, media):
    global ACTIVE
    mode = 'geometry' if geometry else 'events'
    name = f'{c.PREFIX}-{arm}-{stage}-{mode}-{index}'
    directory = c.ROOT / 'containers' / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'data').mkdir()
    if c.owned(name):
        raise RuntimeError('GLOBAL_DUPLICATE_CONTAINER')
    variables = {
        'PYTHONPATH': '/app/src', 'PYTHONDONTWRITEBYTECODE': '1',
        'HARNESS_CONFIG': '/app/config.yaml', 'HARNESS_DEPLOYMENT_SETTINGS_PATH': '/data/deployment-settings.json',
        'HARNESS_AUTOSTART': 'false', 'HARNESS_DRY_RUN': 'false', 'HARNESS_INFERENCE_PROFILE': '',
        'HARNESS_STREAM_URL': '', 'RTSP_URL': '', 'HARNESS_HOST': '0.0.0.0', 'HARNESS_PORT': '8080',
        'INFERENCE_PROVIDER': 'openrouter', 'INFERENCE_MODEL': 'qwen/qwen3-vl-8b-instruct:nitro',
        'SCENE_ENGINE_MODE': 'active', 'PERSON_TRACKING_ENABLED': 'true',
        'RICH_LABEL_ENABLED': 'true', 'RICH_LABEL_BACKEND': 'detector_stream',
        'PERCEPTION_METHOD': 'detector_live', 'DETECTOR_MODEL': arm,
        'DETECTOR_ENDPOINT': 'http://detector-remote:8093/v1/tracks',
        'RICH_LABEL_ENDPOINT': 'http://detector-remote:8093/v1/tracks',
        'RICH_LABEL_MAX_FPS': '1.1', 'SCENE_MAX_IN_FLIGHT': '4', 'SCENE_OBSERVATION_FPS': '1.25',
        'SCENE_MAX_IMAGE_SIDE': '720', 'SCENE_JPEG_QUALITY': '90', 'SCENE_TIMEOUT_SECONDS': '300',
        'SCENE_MAX_TOKENS': '1600', 'FACE_MODEL_ROOT': '/models', 'FACE_GALLERY_PATH': '/data/face-gallery.json',
        'FACE_SAMPLE_FPS': '1.25', 'LOCAL_CAMERA_DISPLAY_ENABLED': 'false', 'RESULT_WEBHOOK_URL': '',
        'ASHLEY_SOURCE_ROOT': '/app', 'ASHLEY_EVENT_DEFINITIONS': '/app/event-definitions.json',
        'ASHLEY_OUTPUT_ROOT': '/run/output', 'ASHLEY_PREVIOUS_RESPONSES': '/run/previous-model-responses.jsonl',
        'ASHLEY_BASELINE_RESPONSE_IDS': '/run/baseline-response-ids.json', 'ASHLEY_KNOWN_COST_LIMIT_USD': '20',
        'ASHLEY_GEOMETRY_ONLY': '1' if geometry else '0',
        'GOODVISION_SAM_SPOOL_DIRECTORY': '/data/detector-spool',
    }
    args = ['run', '-d', '--name', name, '--label', 'experiment=' + c.PREFIX,
            '--network', c.NETWORK, '--restart', 'no', '--init', '--read-only', '--user', '0:0',
            '--no-healthcheck', '--tmpfs', '/tmp:size=256m', '--stop-timeout', '180',
            '-p', '127.0.0.1:18525:8080']
    if not geometry:
        args += ['-e', 'OPENROUTER_API_KEY']
    for key, value in variables.items():
        args += ['-e', key + '=' + value]
    for source, target, readonly in (
        (c.ROOT / 'source/src', '/app/src', True), (c.ROOT / 'source/config.yaml', '/app/config.yaml', True),
        (c.ROOT / 'source/event-definitions.json', '/app/event-definitions.json', True),
        (c.ROOT / 'native_entry.py', '/run/native_entry.py', True), (directory / 'data', '/data', False),
        (c.OUTPUT, '/run/output', False), (c.ROOT / 'previous-model-responses.jsonl', '/run/previous-model-responses.jsonl', True),
        (c.ROOT / 'baseline-response-ids.json', '/run/baseline-response-ids.json', True)):
        args += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    for item in media.values():
        filename = arm + '__' + item['qa_id'] + Path(item['path']).suffix
        args += ['--mount', f"type=bind,src={item['path']},dst=/offline-media/{filename},readonly"]
    args += ['-v', 'goodvision-harness-latency-models:/models:ro', '--entrypoint', 'python', c.IMAGE,
             '-B', '-u', '/run/native_entry.py']
    c.save(directory / 'runtime-config.json', {'environment': variables, 'private_answers_mounted': False,
           'geometry_credentials_mounted': False, 'name': name, 'phase': stage, 'model': arm})
    c.docker(*args)
    ACTIVE = (name, directory)
    deadline = time.time() + 180
    while time.time() < deadline:
        c.stop_loss()
        info = c.owned(name)
        if not info or not info['State']['Running']:
            raise RuntimeError('GLOBAL_NATIVE_STARTUP_FAILED: ' + c.logs(name)[-2000:])
        text = c.logs(name)
        (directory / 'harness.log').write_text(text)
        if 'Application startup complete' in text:
            return name, directory
        time.sleep(2)
    raise RuntimeError('GLOBAL_NATIVE_STARTUP_TIMEOUT')

def finish_video(question, sid, directory, journal, geometry, session):
    previous, changed = None, time.monotonic()
    while True:
        status = c.api('/v1/status', sid=sid)
        c.append(directory / 'status.jsonl', {'recorded_epoch': time.time(), 'status': status})
        if status.get('stream_session_id') != sid:
            raise RuntimeError('GLOBAL_SESSION_ID_MISMATCH')
        rich, drain = status.get('rich_labeling') or {}, status.get('scene_drain') or {}
        target = next((r for r in status.get('postprocessing_runs', []) if r.get('stream_session_id') == sid), None)
        if rich.get('last_error') in {'TrackingJournalFailure', 'OperationalError', 'DatabaseError', 'DiskIOError'}:
            c.save(directory / 'failure-status.json', status)
            raise RuntimeError('GLOBAL_AUDIT_IO_FAILURE: ' + rich['last_error'])
        if (status.get('pipeline_state') in {'error', 'configuration_error'} or status.get('last_error')
                or rich.get('state') == 'failed' or rich.get('recovery_required') or drain.get('state') == 'failed'
                or (target and (target.get('state') == 'failed' or target.get('error')))):
            c.save(directory / 'failure-status.json', status)
            raise RuntimeError('QUESTION_NATIVE_PIPELINE_FAILED')
        # 几何路径由journal的真实EOF和tracking排空证明，事件路径另需原生scene完成。
        outcome = None
        if geometry and journal.exists():
            outcome = extract_question(question, c.OUTPUT, session, geometry_only=True)
        finished = (geometry and outcome and outcome['status'] != 'pending') or (
            not geometry and target and target.get('state') == 'finalized' and target.get('stage') == 'complete')
        if finished:
            c.save(directory / 'final-status.json', status)
            return outcome or extract_question(question, c.OUTPUT, session, geometry_only=False)
        current = (status.get('last_frame_sequence'), rich.get('processed_frames'), status.get('results_published'),
                   drain.get('stage'), target.get('stage') if target else None, journal.stat().st_size if journal.exists() else 0,
                   (c.OUTPUT / 'native-windows.jsonl').stat().st_size if (c.OUTPUT / 'native-windows.jsonl').exists() else 0)
        if current != previous:
            previous, changed = current, time.monotonic()
        elif time.monotonic() - changed > 420:
            c.save(directory / 'stalled-status.json', status)
            raise RuntimeError('QUESTION_NO_PROGRESS_420_SECONDS')
        time.sleep(2)

def run_question(arm, item, question, media, name, container_dir):
    qid = item['qa_id']
    directory = c.ROOT / 'results' / arm / 'questions' / qid
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'outcome.json').exists():
        return False
    if (directory / 'session.json').exists():
        raise RuntimeError('GLOBAL_EXISTING_SESSION_REQUIRES_RECONCILIATION')
    filename = arm + '__' + qid + Path(media[qid]['path']).suffix
    journal = c.OUTPUT / 'records' / filename / 'offline-capture.jsonl'
    sid = None
    session = {'qa_id': qid, 'video_id': question['video_id'], 'detector_model': arm,
               'geometry_only': item['geometry_only'], 'capture_path': str(journal), 'stream_session_id': None,
               'submission_state': 'dispatching', 'started_epoch': time.time()}
    try:
        before = c.api('/v1/status')
        if before.get('processing_lifecycle_busy') or before.get('session_start_pending'):
            raise RuntimeError('GLOBAL_PRIOR_SESSION_STILL_BUSY')
        c.save(directory / 'before-start-status.json', before)
        payload = {'mode': 'file', 'video_path': '/offline-media/' + filename, 'dry_run': False,
                   'analysis_mode': 'multi', 'features': session_features(question), 'camera_layout': 'mono'}
        c.save(directory / 'session-input.json', payload)
        c.save(directory / 'session.json', session)
        started = c.api('/v1/sessions/start', payload, timeout=300)
        sid = started['stream_session_id']
        session.update(stream_session_id=sid, submission_state='started')
        c.save(directory / 'start.json', started)
        c.save(directory / 'session.json', session)
        c.update(phase='native_evaluating', detector_model=arm, geometry_only=item['geometry_only'],
                 current_question=qid, stream_session_id=sid, cohort_phase=item['phase'])
        outcome = finish_video(question, sid, directory, journal, item['geometry_only'], session)
        if outcome['status'] == 'pending':
            raise RuntimeError('GLOBAL_COMPLETION_EVIDENCE_CONTRADICTION')
        session['submission_state'] = 'finalized'
        c.save(directory / 'session.json', session)
    except Exception as error:
        if isinstance(error, OSError) or 'GLOBAL_' in str(error):
            raise
        native_log = c.logs(name)
        detector_log = c.logs(c.PREFIX + '-detector')
        programming_error = re.search(r'(?m)^(?:NameError|ModuleNotFoundError|ImportError|AttributeError|KeyError|TypeError):[^\n]*', native_log + '\n' + detector_log)
        if programming_error:
            c.save(directory / 'programming-stop.json', {'error': programming_error.group(0), 'question_error': str(error)})
            raise RuntimeError('GLOBAL_PROGRAMMING_ERROR: ' + programming_error.group(0)) from error
        outcome = {'qa_id': qid, 'video_id': question['video_id'], 'status': 'pending',
                   'reason': str(error), 'stream_session_id': sid, 'capture_path': str(journal)}
        c.save(directory / 'outcome.json', outcome)
        # 停止本题后归档允许的单题哨兵；未知hash和累计次数仍在共享账本。
        stop_container(name, container_dir)
        stop = c.local_question_stop(sid)
        sentinel = c.OUTPUT / 'stop-loss.json'
        if sentinel.exists():
            if not stop:
                raise RuntimeError('GLOBAL_UNCLASSIFIED_STOP') from error
            sentinel.replace(directory / 'native-stop-loss.json')
        c.save(directory / 'isolation.json', {'same_question_resubmitted': False, 'known_stop': stop,
               'unknown_calls_preserved': c.unknown_for_session(sid), 'stopped_container': name})
        c.stop_loss()
        return True
    outcome.update(detector_model=arm, geometry_only=item['geometry_only'], cohort_phase=item['phase'],
                   elapsed_seconds=time.time() - session['started_epoch'])
    c.save(directory / 'outcome.json', outcome)
    # 等实际服务清理完成后才接受下一题。
    deadline = time.time() + 360
    while time.time() < deadline:
        value = c.api('/v1/status')
        if not value.get('processing_lifecycle_busy') and not value.get('session_start_pending') and value.get('pipeline_state') not in {'starting', 'running', 'stopping', 'draining'}:
            return False
        time.sleep(1)
    raise RuntimeError('GLOBAL_SESSION_CLEANUP_TIMEOUT')

def main():
    c.check_frozen()
    questions = {r['qa_id']: r for r in c.rows(c.ROOT / 'public/questions.jsonl')}
    media = {r['qa_id']: r for r in c.rows(c.ROOT / 'cohort/media-paths.jsonl')}
    queue = c.load(c.ROOT / 'cohort/queue.json')['questions']
    try:
        for stage in ('pilot20', 'remaining'):
            # 公开清单phase的实际名称保存在selection，不按模型成绩分配顺序。
            stage_items = [r for r in queue if (r['phase'] != 'remaining') == (stage == 'pilot20')]
            for geometry in (True, False):
                arms = ('yolo26-s', 'rfdetr-l') if geometry else ('rfdetr-l', 'yolo26-s')
                for arm in arms:
                    selected = [r for r in stage_items if r['geometry_only'] == geometry]
                    if not selected:
                        continue
                    index = 0
                    name, directory = start_container(arm, geometry, stage, index, media)
                    for item in selected:
                        c.stop_loss()
                        restart = run_question(arm, item, questions[item['qa_id']], media, name, directory)
                        if restart and item != selected[-1]:
                            index += 1
                            name, directory = start_container(arm, geometry, stage, index, media)
                    stop_container(name, directory)
            c.save(c.ROOT / f'{stage}-finished.json', {'phase': stage, 'epoch': time.time(),
                   'outcomes': {arm: len(list((c.ROOT / 'results' / arm / 'questions').glob('*/outcome.json'))) for arm in ('yolo26-s', 'rfdetr-l')}})
            if stage == 'pilot20':
                summary = {}
                for arm in ('yolo26-s', 'rfdetr-l'):
                    outcomes = [c.load(p) for p in (c.ROOT / 'results' / arm / 'questions').glob('*/outcome.json')]
                    summary[arm] = {'attempted': len(outcomes), 'complete': sum(r['status'] != 'pending' for r in outcomes),
                                    'pending': [r['qa_id'] for r in outcomes if r['status'] == 'pending'],
                                    'elapsed_seconds': sum(r.get('elapsed_seconds', 0) for r in outcomes)}
                c.save(c.ROOT / 'pilot20-execution-summary.json', summary)
                if any(r['attempted'] != 20 or r['complete'] == 0 for r in summary.values()):
                    raise RuntimeError('GLOBAL_PILOT_EXECUTION_INCOMPLETE_REQUIRES_RECONCILIATION')
    finally:
        if ACTIVE:
            stop_container(*ACTIVE)
