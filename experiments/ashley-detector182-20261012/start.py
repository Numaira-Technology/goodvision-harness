"""持有跨轮双锁并实际核对精确资源后，只启动一次正式控制器。"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dotenv import dotenv_values
from vastai import VastAI
import control as c

def main():
    os.umask(0o077)
    locks = []
    for name in ('benchmark200-recurrent-active.lock', 'ashley-core116-active.lock'):
        handle = (c.ROOT.parent / name).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    if (c.ROOT / 'launch-intent.json').exists() or (c.ROOT / 'launch.json').exists():
        raise RuntimeError('正式启动入口已执行，禁止重跑')
    ready = c.load(c.ROOT / 'ready.json')
    if hashlib.sha256((c.ROOT / 'source-manifest.json').read_bytes()).hexdigest() != ready['manifest_sha256']:
        raise RuntimeError('GLOBAL_MANIFEST_MISMATCH')
    c.check_frozen()
    if c.load(c.ROOT / 'preparation.json')['phase'] != 'prepared_not_started':
        raise RuntimeError('GLOBAL_PREPARATION_STATE_MISMATCH')
    names = c.docker('ps', '-a', '--filter', 'label=experiment=' + c.PREFIX, '--format', '{{.Names}}').stdout.splitlines()
    if names:
        raise RuntimeError('GLOBAL_CURRENT_RESOURCES_ALREADY_EXIST')
    network = json.loads(c.docker('network', 'inspect', c.NETWORK).stdout)[0]
    if network['Name'] != 'ashley116-updated-recurrent-20261004' or network.get('Containers'):
        raise RuntimeError('GLOBAL_DEDICATED_NETWORK_OCCUPIED')
    credentials = {**dotenv_values(c.WORKSPACE / 'goodvision-harness/.env'),
                   **dotenv_values('/Users/pengyuch/.config/goodvision/openrouter.env')}
    for key in ('VAST_API_KEY', 'SAM_SSH_KEY', 'OPENROUTER_API_KEY'):
        if not credentials.get(key) and not os.environ.get(key):
            raise RuntimeError('缺少正式凭据：' + key)
    inventory = VastAI(api_key=credentials.get('VAST_API_KEY') or os.environ['VAST_API_KEY'], retry=1).show_instances() or []
    prior_round = c.ROOT.parent / 'detector-ablation182-resume2-20261012'
    prior_launch = c.load(prior_round / 'launch-intent.json')
    if not c.load(prior_round / 'resources-after.json')['confirmed_released']:
        raise RuntimeError('GLOBAL_FAILED_ROUND_RELEASE_UNCONFIRMED')
    if any(r.get('label') == prior_launch['gpu_label'] for r in inventory):
        raise RuntimeError('GLOBAL_FAILED_ROUND_GPU_REMAINS')
    prior_process = c.command(['ps', '-p', str(c.load(prior_round / 'launch.json')['pid']), '-o', 'command='], check=False).stdout
    if str(prior_round) in prior_process:
        raise RuntimeError('GLOBAL_FAILED_ROUND_CONTROLLER_REMAINS')
    if time.time() >= prior_launch['deadline_epoch']:
        raise RuntimeError('GLOBAL_ORIGINAL_AUTHORIZED_DEADLINE')
    label = 'goodvision-' + c.PREFIX
    if any(r.get('label') == label for r in inventory):
        raise RuntimeError('GLOBAL_CURRENT_GPU_ALREADY_EXISTS')
    old = c.ROOT.parent / 'main-event-supplement30-ashley1-resume10-20261009'
    released = c.load(old / 'resources-after.json')
    if not released['confirmed_released'] or any(r.get('id') in released.get('owned_ids', []) or r.get('label') in released.get('owned_labels', []) for r in inventory):
        raise RuntimeError('GLOBAL_PRIOR_RESOURCE_REMAINS')
    for pid in (c.load(old / 'launch.json')['pid'], c.load(old / 'run.json').get('ashley_pid')):
        if pid:
            process = c.command(['ps', '-p', str(pid), '-o', 'command='], check=False).stdout
            if str(old) in process:
                raise RuntimeError('GLOBAL_PRIOR_CONTROLLER_STILL_RUNNING')
    c.save(c.ROOT / 'resources-before.json', {'checked_epoch': time.time(), 'network': c.NETWORK,
           'containers': [], 'inventory': [{k: r.get(k) for k in ('id', 'label', 'actual_status')} for r in inventory]})
    started = prior_launch['started_epoch']
    actual_launch = time.time()
    deadline = prior_launch['deadline_epoch']
    c.save(c.ROOT / 'launch-intent.json', {'started_epoch': started, 'actual_launch_epoch': actual_launch, 'deadline_epoch': deadline,
           'prefix': c.PREFIX, 'gpu_label': label, 'questions_per_arm': 182})
    with (c.ROOT / 'controller.log').open('xb', buffering=0) as log:
        child = subprocess.Popen(['/usr/bin/caffeinate', '-i', sys.executable, '-B', '-u', str(c.ROOT / 'driver.py')],
            cwd=c.ROOT, env=dict(os.environ, DETECTOR_LOCK_FDS=','.join(str(h.fileno()) for h in locks)),
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
            pass_fds=tuple(h.fileno() for h in locks))
    c.save(c.ROOT / 'launch.json', {'pid': child.pid, 'started_epoch': started, 'actual_launch_epoch': actual_launch, 'deadline_epoch': deadline,
           'prefix': c.PREFIX, 'gpu_label': label, 'status': 'controller_started_not_model_ready'})
    print(json.dumps(c.load(c.ROOT / 'launch.json')))

if __name__ == '__main__':
    main()
