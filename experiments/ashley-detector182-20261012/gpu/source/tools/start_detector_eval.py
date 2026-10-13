"""正式评测专用：单GPU、精确资源归属、日志启动标记，无健康或模型探针。"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from decimal import Decimal
from pathlib import Path

from remote_helpers import (
    CreationRejected, create_from_offers, creation_diagnostic, release_instance,
    required, run, stop_process,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data")
REMOTE = "/workspace/goodvision-detector-eval"
PORT = 8093
LABEL = os.environ.get("DETECTOR_INSTANCE_LABEL", "goodvision-detector-ablation182-20261012")
CAP = Decimal("10")
DISK_GB = 30
FILES = ("app.py", "detectors.py", "tracker.py", "requirements.txt", "download_weights.py")

SETUP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update
apt-get install -y --no-install-recommends python3.12-venv python3-pip build-essential ca-certificates libgomp1 libglib2.0-0 libgl1
python3.12 -m venv {REMOTE}/env
{REMOTE}/env/bin/python -m pip install --no-cache-dir pip==25.1.1 setuptools==80.4.0 wheel==0.45.1
{REMOTE}/env/bin/python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
{REMOTE}/env/bin/python -m pip install --no-cache-dir -r {REMOTE}/worker/requirements.txt
{REMOTE}/env/bin/python -m pip list --format=json > {REMOTE}/records/installed-packages.json
{REMOTE}/env/bin/python {REMOTE}/worker/download_weights.py {REMOTE}/models
"""


def save(name, value):
    target = DATA / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)


def log(event, **values):
    record = {"event": event, "recorded_epoch": time.time(), **values}
    text = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(text, flush=True)
    with (DATA / "controller.jsonl").open("a") as output:
        output.write(text + "\n")
        output.flush()


def inventory(vast):
    result = vast.show_instances()
    if not isinstance(result, list):
        raise RuntimeError("供应商实例列表结构未知")
    return result


def resource_identities(rows):
    return [{key: row.get(key) for key in ('id', 'label', 'actual_status', 'host_id', 'machine_id')} for row in rows]


def await_deployment_address(vast, instance_id, deadline):
    """只读取正式资源状态；首次SSH执行的是实际部署目录创建。"""
    previous = None
    until = min(deadline, time.time() + 20 * 60)
    while time.time() < until:
        record = vast.show_instance(id=instance_id)
        if not isinstance(record, dict) or int(record.get("id", -1)) != instance_id:
            raise RuntimeError("正式实例记录缺失")
        if record.get("label") != LABEL:
            raise RuntimeError("正式实例label发生变化，拒绝连接")
        current = (record.get("actual_status"), record.get("status_msg"))
        if current != previous:
            log("provider_deployment_state", instance_id=instance_id, status=current)
            previous = current
        if record.get("actual_status") in {"offline", "exited", "stopped", "frozen"}:
            raise RuntimeError("供应商容器在SSH部署前停止")
        if record.get("actual_status") == "running":
            ssh_ports = (record.get("ports") or {}).get("22/tcp", [])
            if record.get("public_ipaddr") and ssh_ports:
                return str(record["public_ipaddr"]), int(ssh_ports[0]["HostPort"])
            if record.get("ssh_host") and record.get("ssh_port"):
                return str(record["ssh_host"]), int(record["ssh_port"])
        time.sleep(5)
    raise RuntimeError("正式资源部署地址等待超时")


def charge_snapshot(vast, instance_id, started):
    rows = []
    token = None
    seen_tokens = set()
    for _ in range(20):
        params = {
            "charges": True, "charge_type": ["instance"], "format": "tree",
            "start_date": int(started // 86400 * 86400),
            "end_date": int(time.time() // 86400 * 86400 + 86400),
            "limit": 100, "latest_first": True,
        }
        if token is not None:
            params["next_token"] = token
        response = vast.show_invoices_v1(**params)
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise RuntimeError("供应商收费结果结构未知")
        rows.extend(row for row in response["results"] if row.get("source") == f"instance-{instance_id}")
        token = response.get("next_token")
        if not token:
            break
        if token in seen_tokens:
            raise RuntimeError("供应商收费分页重复")
        seen_tokens.add(token)
    else:
        raise RuntimeError("供应商收费分页超出边界，费用保持未知")
    values = []
    for row in rows:
        value = Decimal(str(row.get("amount")))
        if not value.is_finite() or value < 0:
            raise RuntimeError("供应商收费金额无效")
        values.append(value)
    # 空账单仅表示尚未观察到收费行，不能算作0美元。
    amount = sum(values, Decimal(0)) if rows else None
    return {
        "checked_epoch": time.time(), "instance_id": instance_id,
        "source": "Vast show_invoices_v1 /api/v0/charges/", "rows": rows,
        "actual_charge_usd": str(amount) if amount is not None else None,
        "charge_status": "observed_provider_snapshot" if rows else "unknown_no_charge_rows",
        "final_invoice": False,
    }


def create_deployment_directories(ssh, deadline):
    """实际部署mkdir可幂等重试；不增加SSH探针或模型预热。"""
    until = min(deadline, time.time() + 180)
    attempt = 0
    while time.time() < until:
        attempt += 1
        try:
            result = subprocess.run([*ssh, f"mkdir -p {REMOTE}/worker {REMOTE}/models {REMOTE}/records"],
                                    capture_output=True, text=True, timeout=min(45, max(1, until - time.time())))
        except subprocess.TimeoutExpired:
            log('deployment_directory_retry', attempt=attempt, reason='ssh_command_timeout')
        else:
            if result.returncode == 0:
                log('deployment_directories_created', attempt=attempt)
                return
            diagnostic = creation_diagnostic({'message': result.stderr})
            retryable = result.returncode == 255 and any(value in result.stderr.lower() for value in (
                'connection closed', 'connection refused', 'connection reset', 'connection timed out',
                'connection timeout', 'kex_exchange_identification'))
            if any(value in result.stderr.lower() for value in ('permission denied', 'host key verification failed', 'remote host identification has changed')):
                retryable = False
            log('deployment_directory_failure', attempt=attempt, returncode=result.returncode, diagnostic=diagnostic, retryable=retryable)
            if not retryable:
                raise RuntimeError('正式部署目录创建失败：' + diagnostic)
        time.sleep(min(5, max(0, until - time.time())))
    raise RuntimeError('正式部署目录创建在180秒有界恢复后仍未完成')


def billing_loop_body(api_key, instance_id, started, deadline, offer_dph, prior, stop):
    from vastai import VastAI
    vast = VastAI(api_key=api_key, retry=1)
    vast.client.timeout = (5, 20)
    next_charge = 0.0
    current_known_floor = Decimal(0)
    prior_resources = json.loads((ROOT / 'prior-resources.json').read_text())['instances']
    while not stop.is_set():
        reason = None
        if (DATA / "STOP").exists():
            reason = "parent_stop_file"
        elif time.time() >= deadline:
            reason = "authorized_deadline_reached"
        if time.monotonic() >= next_charge:
            next_charge = time.monotonic() + 60
            prior_snapshots = []
            prior_known = prior
            prior_estimate = prior
            for resource in prior_resources:
                try:
                    snapshot = charge_snapshot(vast, resource['instance_id'], resource['started_epoch'])
                except Exception as error:
                    snapshot = {'instance_id': resource['instance_id'], 'actual_charge_usd': None,
                                'charge_status': 'unknown_read_failed', 'error': creation_diagnostic({'message': str(error)})}
                previous_known = Decimal(resource['minimum_observed_charge_usd'])
                observed = snapshot.get('actual_charge_usd')
                known = max(previous_known, Decimal(observed)) if observed is not None else previous_known
                resource['minimum_observed_charge_usd'] = str(known)
                prior_known += known
                elapsed_estimate = Decimal(resource['quoted_dph']) * Decimal(str((resource['stopped_epoch'] - resource['started_epoch']) / 3600))
                prior_estimate += max(known, elapsed_estimate)
                snapshot['known_charge_floor_usd'] = str(known)
                prior_snapshots.append(snapshot)
            try:
                record = charge_snapshot(vast, instance_id, started)
                actual = record["actual_charge_usd"]
                if actual is not None:
                    current_known_floor = max(current_known_floor, Decimal(actual))
            except Exception as error:
                record = {
                    "checked_epoch": time.time(), "instance_id": instance_id,
                    "actual_charge_usd": None, "charge_status": "unknown_read_failed",
                    "error": creation_diagnostic({"message": str(error)}),
                }
            estimate = prior_estimate + offer_dph * Decimal(str(max(0, time.time() - started) / 3600))
            if prior_known + current_known_floor >= CAP:
                reason = "provider_actual_gpu_charge_cap_reached"
            record.update({
                "prior_actual_charge_usd": str(prior_known), "cap_usd": str(CAP),
                "prior_instance_snapshots": prior_snapshots,
                "current_known_charge_floor_usd": str(current_known_floor),
                "cumulative_known_charge_usd": str(prior_known + current_known_floor),
                "rental_estimate_usd_not_actual": str(estimate),
                "estimate_excludes_unreported_transfer_and_other_fees": True,
            })
            save("gpu-cost.json", record)
            log("gpu_cost_snapshot", **record)
            # 附加保守止损，不把报价估算冒充实际账单。
            if estimate >= CAP:
                reason = "quoted_rental_estimate_cap_reached"
        if reason:
            save("stop-reason.json", {"reason": reason, "recorded_epoch": time.time()})
            log("budget_or_control_stop", reason=reason)
            os.kill(os.getpid(), signal.SIGTERM)
            return
        stop.wait(1)


def billing_loop(*args):
    # 费用/截止守护本身失败时终止主控制器；不允许仅daemon线程静默退出。
    try:
        billing_loop_body(*args)
    except BaseException as error:
        try:
            os.write(2, ('GLOBAL_GPU_BUDGET_WATCHDOG_FAILED: ' + type(error).__name__ + '\n').encode())
        finally:
            os.kill(os.getpid(), signal.SIGTERM)


def cleanup(vast, instance_id, attempted, creation_unknown, worker, label):
    result = {
        "recorded_epoch": time.time(), "label": label, "instance_id": instance_id,
        "creation_attempted": attempted, "creation_outcome_unknown": creation_unknown,
        "confirmed_released": False, "errors": [], "matched_instances_after": None,
    }
    try:
        stop_process(worker)
    except Exception as error:
        result["errors"].append(creation_diagnostic({"message": str(error)}))
    if vast is None:
        result["confirmed_released"] = not attempted
        save("resources-after.json", result)
        return result
    vast.client.timeout = (5, 20)
    try:
        before = inventory(vast)
        owned = [row for row in before if row.get("label") == label]
        result["matched_instances_before"] = resource_identities(owned)
        if not attempted:
            # 预先存在同label只意味着阻止重复创建，绝不转化为释放授权。
            result["matched_instances_after"] = resource_identities(owned)
            result["actual_inventory_read_epoch"] = time.time()
            result["confirmed_released"] = not owned
            result["no_creation_by_this_controller"] = True
            save("resources-after.json", result)
            return result
        if instance_id is not None:
            same_id = [row for row in before if int(row.get("id", -1)) == instance_id]
            if any(row.get("label") != label for row in same_id):
                raise RuntimeError("实例ID的label不符，禁止释放该资源")
        if len(owned) > 1:
            raise RuntimeError("本轮精确label出现多个实例，需父控核对，禁止扩大释放范围")
        if owned:
            owned_id = int(owned[0]["id"])
            if instance_id is not None and owned_id != instance_id:
                raise RuntimeError("label与创建回执ID冲突")
            instance_id = owned_id
            result["instance_id"] = owned_id
            release_instance(vast, owned_id)
        for _ in range(6):
            after = inventory(vast)
            matching = [row for row in after if row.get("label") == label or (
                instance_id is not None and int(row.get("id", -1)) == instance_id
            )]
            result["matched_instances_after"] = resource_identities(matching)
            result["actual_inventory_read_epoch"] = time.time()
            if not matching:
                # 丢失创建响应且未找到资源时，后续仍可能出现，必须保持未知。
                result["confirmed_released"] = not (creation_unknown and instance_id is None)
                break
            time.sleep(2)
    except Exception as error:
        result["errors"].append(creation_diagnostic({"message": str(error)}))
    save("resources-after.json", result)
    log("resource_release_result", **result)
    return result


def main():
    from vastai import VastAI
    DATA.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"goodvision-detector-[a-z0-9-]{8,80}", LABEL):
        raise RuntimeError("无效的本轮精确label")
    started = time.time()
    deadline = float(os.environ.get("DETECTOR_DEADLINE_EPOCH", str(started + 24 * 3600)))
    prior = Decimal(os.environ.get("DETECTOR_GPU_PRIOR_USD", "0"))
    if not math.isfinite(deadline) or deadline <= started or deadline > started + 24 * 3600:
        raise RuntimeError("正式截止必须在未来且不超过24小时")
    if not prior.is_finite() or prior < 0 or prior >= CAP:
        raise RuntimeError("累计GPU实际费用无效或已达上限")
    # 一个/data仅允许一次启动。创建不明、部署失败均不能覆盖历史再次租机。
    with (DATA / "launch-intent.json").open("x") as output:
        json.dump({"label": LABEL, "started_epoch": started, "deadline_epoch": deadline}, output)
        output.flush()
        os.fsync(output.fileno())
    deployment = {
        "state": "preparing", "label": LABEL, "started_epoch": started,
        "deadline_epoch": deadline, "instance_id": None,
        "endpoint": "http://detector-remote:8093/v1/tracks",
        "startup_evidence": "uvicorn_log_only", "model_inference_confirmed": False,
    }
    save("deployment.json", deployment)
    vast = None
    instance_id = None
    attempted = False
    creation_unknown = False
    worker = None
    bill_thread = None
    stop_billing = threading.Event()
    ssh_directory = None
    exit_code = 1
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    for item in signals:
        signal.signal(item, interrupt)
    try:
        api_key = required("VAST_API_KEY")
        key_source = Path(required("SAM_SSH_KEY"))
        if not key_source.is_file():
            raise RuntimeError("SAM_SSH_KEY文件不存在")
        ssh_directory = tempfile.TemporaryDirectory(prefix="detector-eval-ssh-")
        identity = Path(ssh_directory.name) / "identity"
        shutil.copyfile(key_source, identity)
        identity.chmod(0o600)
        options = [
            "-i", str(identity), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
            "-o", f"UserKnownHostsFile={Path(ssh_directory.name) / 'known_hosts'}",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        ]
        vast = VastAI(api_key=api_key, retry=1)
        if any(row.get("label") == LABEL for row in inventory(vast)):
            raise RuntimeError("精确label已有实例，禁止重复租机；交父控按历史记录恢复")
        offers = vast.search_offers(
            query=f"gpu_name=RTX_4090 num_gpus=1 rentable=true direct_port_count>=1 cuda_max_good>=12.6 disk_space>={DISK_GB} geolocation!=CN",
            type="on-demand", order="dph_total", storage=float(DISK_GB), limit=20,
        )
        offers = [row for row in (offers or []) if (
            str(row.get("host_id")) != "151177" and str(row.get("machine_id")) != "34105"
            and row.get("public_ipaddr") != "89.166.84.205"
            and math.isfinite(float(row.get("dph_total", "inf")))
            and 0 < float(row["dph_total"]) <= 0.75
        )]
        if not offers:
            raise RuntimeError("没有合格的一GPU报价；未租机")
        offer = min(offers, key=lambda row: float(row["dph_total"]))
        deployment["offer"] = offer
        save("deployment.json", deployment)
        attempted = True
        creation_unknown = True
        deployment["state"] = "creation_attempted"
        save("deployment.json", deployment)
        try:
            # 单一offer、SDK retry=1；创建未知不会发送第二次创建。
            instance_id = create_from_offers(vast, [offer], disk_gb=DISK_GB, label=LABEL)
        except CreationRejected:
            creation_unknown = False
            raise
        creation_unknown = False
        deployment.update(state="instance_created", instance_id=instance_id)
        save("deployment.json", deployment)
        bill_thread = threading.Thread(target=billing_loop, args=(
            api_key, instance_id, started, deadline, Decimal(str(offer["dph_total"])), prior, stop_billing,
        ), daemon=True)
        bill_thread.start()
        host, ssh_port = await_deployment_address(vast, instance_id, deadline)
        ssh = ["ssh", *options, "-p", str(ssh_port), f"root@{host}"]
        deployment.update(state="installing", ssh_host=host, ssh_port=ssh_port)
        save("deployment.json", deployment)
        create_deployment_directories(ssh, deadline)
        run(["scp", *options, "-P", str(ssh_port), *(
            str(ROOT / "workers" / "detector" / name) for name in FILES
        ), f"root@{host}:{REMOTE}/worker/"])
        run([*ssh, "bash -s"], input=SETUP, timeout=45 * 60)
        (DATA / "weights").mkdir(exist_ok=True)
        run(["scp", *options, "-P", str(ssh_port),
             f"root@{host}:{REMOTE}/models/weights.json",
             f"root@{host}:{REMOTE}/models/yolo26s.pt",
             f"root@{host}:{REMOTE}/models/rf-detr-large.pth", str(DATA / "weights")], timeout=30 * 60)
        run(["scp", *options, "-P", str(ssh_port),
             f"root@{host}:{REMOTE}/records/installed-packages.json", str(DATA)])
        deployment.update(state="starting_application", weights=json.loads((DATA / "weights/weights.json").read_text()))
        save("deployment.json", deployment)
        command = (
            f"cd {REMOTE}/worker && exec env PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false "
            f"DETECTOR_WEIGHTS_DIR={REMOTE}/models DETECTOR_RUNTIME_RECORD={REMOTE}/records/runtime.jsonl "
            f"{REMOTE}/env/bin/python -m uvicorn app:app --host 127.0.0.1 --port {PORT} --workers 1"
        )
        worker = subprocess.Popen([
            *ssh[:-1], "-C", "-o", "ExitOnForwardFailure=yes", "-L",
            f"0.0.0.0:{PORT}:127.0.0.1:{PORT}", ssh[-1], command,
        ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        with (DATA / "worker.log").open("a") as worker_log:
            for line in worker.stdout:
                worker_log.write(line)
                worker_log.flush()
                print(line, end="", flush=True)
                if "Application startup complete." in line:
                    deployment.update(state="application_started", application_started_epoch=time.time())
                    save("deployment.json", deployment)
                    log("application_started", label=LABEL, instance_id=instance_id, port=PORT,
                        model_inference_confirmed=False, evidence="Application startup complete.")
                if line.startswith("DETECTOR_RUNTIME "):
                    with (DATA / "model-runtime.jsonl").open("a") as records:
                        records.write(line[len("DETECTOR_RUNTIME "):])
                        records.flush()
        worker.wait()
        raise RuntimeError(f"正式worker/SSH已退出，returncode={worker.returncode}")
    except KeyboardInterrupt:
        deployment["stop_reason"] = "signal_or_budget_or_parent_stop"
        exit_code = 130
    except Exception as error:
        deployment["error"] = creation_diagnostic({"message": str(error)})
        log("deployment_stopped", error=deployment["error"])
    finally:
        for item in signals:
            signal.signal(item, signal.SIG_IGN)
        stop_billing.set()
        deployment["state"] = "stopping"
        save("deployment.json", deployment)
        resource = cleanup(vast, instance_id, attempted, creation_unknown, worker, LABEL)
        deployment.update(state="released" if resource["confirmed_released"] else "release_unconfirmed",
                          stopped_epoch=time.time())
        save("deployment.json", deployment)
        if ssh_directory is not None:
            ssh_directory.cleanup()
        if not resource["confirmed_released"]:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
