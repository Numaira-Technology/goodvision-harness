"""单GPU双检测器正式评测控制器。finally仅释放本轮精确资源。"""
import json
import os
import signal
import time
from dotenv import dotenv_values
from vastai import VastAI
import control as c

def start_gpu():
    directory = c.ROOT / 'gpu/output'
    directory.mkdir(parents=True, exist_ok=True)
    launch = c.load(c.ROOT / 'launch-intent.json')
    c.docker('run', '-d', '--name', c.PREFIX + '-detector', '--label', 'experiment=' + c.PREFIX,
             '--network', c.NETWORK, '--network-alias', 'detector-remote', '--restart', 'no', '--init',
             '--read-only', '--no-healthcheck', '--tmpfs', '/tmp:size=128m', '--stop-timeout', '240',
             '-e', 'VAST_API_KEY', '-e', 'SAM_SSH_KEY=/run/secrets/sam_ssh_key',
             '-e', 'DETECTOR_INSTANCE_LABEL=' + launch['gpu_label'],
             '-e', 'DETECTOR_DEADLINE_EPOCH=' + str(launch['deadline_epoch']),
             '-e', 'DETECTOR_GPU_PRIOR_USD=0', '-e', 'PYTHONDONTWRITEBYTECODE=1',
             '-v', c.ENV['SAM_SSH_KEY'] + ':/run/secrets/sam_ssh_key:ro',
             '-v', str(c.ROOT / 'gpu/source') + ':/app:ro', '-v', str(directory) + ':/data',
             '--entrypoint', 'python', c.GPU_IMAGE, '-B', '-u', '/app/tools/start_detector_eval.py')
    c.update(phase='preparing_detector')
    deadline = time.time() + 7200
    while time.time() < deadline:
        c.stop_loss()
        (directory / 'startup.log').write_text(c.logs(c.PREFIX + '-detector'))
        path = directory / 'deployment.json'
        if path.exists():
            state = c.load(path)
            if state.get('state') == 'application_started':
                c.update(phase='detector_application_started', deployment=state,
                         notice='应用启动日志已就绪，尚不表示任何模型已完成正式推理')
                return
        time.sleep(5)
    raise RuntimeError('GLOBAL_DEPLOYMENT_TIMEOUT')

def cleanup():
    errors = []
    local_confirmed = True
    try:
        names = c.docker('ps', '-a', '--filter', 'label=experiment=' + c.PREFIX, '--format', '{{.Names}}').stdout.splitlines()
    except Exception as error:
        names = [c.PREFIX + '-detector']
        local_confirmed = False
        errors.append('local_enumeration: ' + str(error))
    # 先停止原生调用方，再停止GPU控制器，使其finally按精确label释放租用。
    for name in sorted(names, key=lambda n: n == c.PREFIX + '-detector'):
        try:
            info = c.owned(name)
            if info and info['State']['Running']:
                c.docker('stop', '--time', '240', name, timeout=300)
            (c.ROOT / 'logs').mkdir(exist_ok=True)
            (c.ROOT / 'logs' / (name + '.log')).write_text(c.logs(name))
        except Exception as error:
            local_confirmed = False
            errors.append(str(error))
    # 本地Docker失联不阻止独立执行供应商侧精确释放。
    label = 'goodvision-' + c.PREFIX
    child_path = c.ROOT / 'gpu/output/resources-after.json'
    attempted_path = c.ROOT / 'gpu/output/deployment.json'
    child, attempted = None, {}
    metadata_confirmed = True
    for path, kind in ((child_path, 'child'), (attempted_path, 'attempted')):
        try:
            value = c.load(path) if path.exists() else None
            if kind == 'child':
                child = value
            else:
                attempted = value or {}
        except Exception as error:
            metadata_confirmed = False
            errors.append(kind + '_metadata: ' + str(error))
    known_ids = {int(value) for value in (attempted.get('instance_id'), (child or {}).get('instance_id')) if value is not None}
    remote_confirmed, remaining = False, []
    try:
        vast = VastAI(api_key=c.ENV['VAST_API_KEY'], retry=1)
        vast.client.timeout = (5, 20)
        inventory = vast.show_instances() or []
        remaining = [r for r in inventory if r.get('label') == label]
        for row in remaining:
            try:
                if len(remaining) > 1 or (known_ids and int(row['id']) not in known_ids):
                    raise RuntimeError('GLOBAL_RESOURCE_ID_LABEL_CONFLICT')
                vast.destroy_instance(id=int(row['id']))
            except Exception as error:
                errors.append(str(error))
        remaining = [r for r in (vast.show_instances() or []) if r.get('label') == label or int(r['id']) in known_ids]
        remote_confirmed = not remaining
    except Exception as error:
        errors.append('remote_release: ' + str(error))
    states = {}
    for name in names:
        try:
            info = c.owned(name)
            if info:
                states[name] = info['State']
        except Exception as error:
            local_confirmed = False
            errors.append('local_state: ' + str(error))
    network = {}
    try:
        network = json.loads(c.docker('network', 'inspect', c.NETWORK).stdout)[0]
    except Exception as error:
        local_confirmed = False
        errors.append('network_state: ' + str(error))
    unresolved_creation = bool(child and child.get('creation_outcome_unknown') and not child.get('instance_id'))
    if not child and attempted.get('state') == 'creation_attempted' and not attempted.get('instance_id'):
        unresolved_creation = True
    confirmed = metadata_confirmed and local_confirmed and remote_confirmed and not unresolved_creation and not any(s['Running'] for s in states.values()) and not network.get('Containers')
    c.save(c.ROOT / 'resources-after.json', {'checked_epoch': time.time(), 'confirmed_released': confirmed,
           'gpu_label': label, 'known_instance_ids': sorted(known_ids), 'remaining': [{k: r.get(k) for k in ('id', 'label', 'actual_status')} for r in remaining],
           'containers': states, 'network_containers': network.get('Containers', {}), 'cleanup_errors': errors,
           'local_confirmed': local_confirmed, 'remote_confirmed': remote_confirmed, 'metadata_confirmed': metadata_confirmed,
           'creation_outcome_unresolved': unresolved_creation, 'child_cleanup': child})
    return 'confirmed_released' if confirmed else 'release_unconfirmed'

def main():
    locks = [os.fdopen(int(fd), 'a') for fd in os.environ['DETECTOR_LOCK_FDS'].split(',')]
    credentials = {**dotenv_values(c.WORKSPACE / 'goodvision-harness/.env'),
                   **dotenv_values('/Users/pengyuch/.config/goodvision/openrouter.env')}
    for key in ('VAST_API_KEY', 'SAM_SSH_KEY', 'OPENROUTER_API_KEY'):
        c.ENV[key] = credentials.get(key) or os.environ[key]
    def interrupted(signum, frame):
        raise KeyboardInterrupt('controller signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    c.update(phase='starting', controller_pid=os.getpid(), started_epoch=c.load(c.ROOT / 'launch-intent.json')['started_epoch'])
    error = None
    try:
        c.check_frozen()
        start_gpu()
        import run_batch
        run_batch.main()
        c.update(phase='both_arms_finished_pending_report')
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        c.save(c.ROOT / 'failure.json', error)
        c.update(phase='stopping', error=error)
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        try:
            release = cleanup()
        except BaseException as exc:
            release = 'release_unconfirmed'
            c.save(c.ROOT / 'cleanup-error.json', {'error': str(exc)})
        c.update(phase='stopped', cleanup=release, error=error)
        c.save(c.ROOT / 'round-outcome.json', {'finished_epoch': time.time(), 'error': error, 'cleanup': release,
               'outcome_counts': {arm: len(list((c.ROOT / 'results' / arm / 'questions').glob('*/outcome.json'))) for arm in ('yolo26-s', 'rfdetr-l')}})
        for handle in locks:
            handle.close()

if __name__ == '__main__':
    main()
