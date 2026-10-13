"""仅复用历史控制器的创建、进程与释放帮助函数；无探针入口。"""
from __future__ import annotations
import os
import re
import signal
import subprocess
from pathlib import Path

IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
SSH_ONSTART = """set -eu
chown root:root /root/.ssh /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
"""

def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in the repository .env or your terminal.")
    return value


class CreationRejected(RuntimeError):
    """Vast explicitly refused creation; no contract needs reconciliation."""


def creation_diagnostic(payload, *, status: int | None = None) -> str:
    """Keep the provider's error fields without logging requests or credentials."""
    fields = []

    def collect(value, depth=0):
        if not isinstance(value, dict) or depth > 2:
            return
        for name in ("code", "error_code", "error", "message", "msg", "detail"):
            item = value.get(name)
            if isinstance(item, dict):
                collect(item, depth + 1)
            elif isinstance(item, (str, int)) and not isinstance(item, bool):
                fields.append(f"{name}={item}")

    collect(payload)
    detail = "; ".join(fields) or "No structured error details returned"
    # Redact before truncating so a clipped credential cannot escape replacement.
    for name in ("VAST_API_KEY", "HF_TOKEN", "SAM_SSH_KEY"):
        secret = os.environ.get(name, "")
        if secret:
            detail = detail.replace(secret, "[redacted]")
    detail = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|$)",
        "[redacted private key]", detail, flags=re.DOTALL,
    )
    detail = re.sub(
        r"(?i)\b(?:bearer\s+|hf_)[a-z0-9._~+/=-]+", "[redacted]", detail,
    )
    detail = re.sub(
        r"(?i)\b(api[_ -]?key|token|password|secret|authorization)\s*[:=]\s*"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1=[redacted]", detail,
    )
    detail = " ".join(detail.split())[:1000]
    return f"HTTP {status}: {detail}" if status is not None else detail


def offer_was_unavailable(payload) -> bool:
    """Only recognize explicit capacity/offer refusals, never generic HTTP 400s."""
    if not isinstance(payload, dict):
        return False
    # Vast把已下架报价的精确no_such_ask消息包装成invalid_args。
    # 只识别本次实际返回的完整结构；普通invalid_args仍不得切换租机。
    if (payload.get("error") == "invalid_args"
            and payload.get("code") is None and payload.get("error_code") is None
            and isinstance(payload.get("msg"), str)
            and re.fullmatch(
                r"error 404/3603: no_such_ask Instance type by id [0-9]+ "
                r"is not available\.\.?", payload["msg"].strip())):
        return True
    codes = {
        "no_such_ask", "no_such_offer", "ask_not_found", "offer_not_found",
        "ask_unavailable", "offer_unavailable", "machine_unavailable",
        "offer_already_rented", "ask_already_rented", "offer_taken",
    }
    for name in ("code", "error_code"):
        value = payload.get(name)
        if isinstance(value, str) and value.lower() not in codes:
            return False
    error = payload.get("error")
    if (
        isinstance(error, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", error)
        and error.lower() not in codes
    ):
        return False
    if isinstance(payload.get("error"), dict):
        return offer_was_unavailable(payload["error"])
    for name in ("code", "error_code", "error"):
        value = payload.get(name)
        if isinstance(value, str) and value.lower() in codes:
            return True
    # Keep the unavailable phrase adjacent to its subject: "image not found"
    # or "invalid parameters for offer 123" must not trigger another rental.
    message = creation_diagnostic(payload).lower()
    return re.search(
        r"\b(?:offer|ask|machine)(?:\s+(?:id\s+)?#?\d+)?\s+"
        r"(?:(?:is|was|has been)\s+)?(?:not found|does not exist|no longer exists|"
        r"not available|unavailable|no longer available|already (?:rented|taken))\b",
        message,
    ) is not None


def create_from_offers(
    vast, offers: list[dict], *, disk_gb: int, label: str,
    attempted_machines: set[str] | None = None,
) -> int:
    """Share a three-machine budget across creation and startup failures."""
    seen_machines = attempted_machines if attempted_machines is not None else set()
    seen_offers = set()
    attempts = len(seen_machines)
    for offer in sorted(offers, key=lambda item: float(item["dph_total"])):
        offer_id = int(offer["id"])
        # Without a machine identity we cannot prove that another unknown offer
        # is a different host, so try at most one such candidate.
        machine = str(offer.get("machine_id") or "unknown")
        if offer_id in seen_offers or machine in seen_machines:
            continue
        if attempts == 3:
            break
        seen_offers.add(offer_id)
        seen_machines.add(machine)
        attempts += 1
        print(
            f"Selected offer: {offer_id}; location: {offer.get('geolocation') or 'unknown'}; "
            f"estimated total hourly price: "
            f"${float(offer['dph_total']):.4f}; candidate {attempts}/3", flush=True,
        )
        print(f"Creating instance with label: {label}", flush=True)
        status = None
        try:
            created = vast.create_instance(
                id=offer_id, image=IMAGE, disk=disk_gb,
                label=label, ssh=True, direct=True, cancel_unavail=True,
                onstart_cmd=SSH_ONSTART,
            )
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            try:
                created = response.json() if response is not None else None
            except (ValueError, TypeError):
                created = None
            detail = creation_diagnostic(created, status=status)
            # A timeout, server error or lost response may hide a successful
            # creation. Stop here; main reconciles this unique label before exit.
            if not isinstance(status, int) or not 400 <= status < 500 or status == 408:
                raise RuntimeError(
                    f"Creation outcome unknown for offer {offer_id} "
                    f"({type(error).__name__}); {detail}"
                ) from None
        else:
            contract = (
                created.get("new_contract") if isinstance(created, dict) else None
            )
            if (
                isinstance(created, dict) and created.get("success") is True
                and isinstance(contract, int) and not isinstance(contract, bool)
                and contract > 0
            ):
                return contract
            if not isinstance(created, dict) or created.get("success") is not False:
                raise RuntimeError(
                    f"Creation outcome unknown for offer {offer_id}: "
                    "invalid creation response; "
                    + creation_diagnostic(created)
                )
        detail = creation_diagnostic(created, status=status)
        if isinstance(created, dict) and created.get("new_contract"):
            raise RuntimeError(
                f"Creation outcome unknown for offer {offer_id}: "
                f"refusal includes a contract; {detail}"
            )
        if status in {None, 400, 404, 409, 410} and offer_was_unavailable(created):
            print(
                f"Offer {offer_id} was refused: {detail}; skipping this machine.",
                flush=True,
            )
            continue
        raise CreationRejected(
            f"Creation refused for offer {offer_id}; {detail}. No instance created."
        )
    raise CreationRejected(
        f"No instance created: no untried offers remain after {attempts} machine attempts "
        "(at most 3 machines per launch)."
    )



def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(command: list[str], *, input: str | None = None, timeout: int = 120) -> None:
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        text=True, start_new_session=True,
    )
    try:
        process.communicate(input=input, timeout=timeout)
        if process.returncode:
            raise RuntimeError(f"{Path(command[0]).name} exited with status {process.returncode}.")
    finally:
        stop_process(process)



def release_instance(vast, instance_id: int) -> None:
    """Require confirmed cleanup before another paid rental can be created."""
    print(f"Destroying instance: {instance_id}", flush=True)
    previous_timeout = vast.client.timeout
    vast.client.timeout = (5, 20)
    try:
        result = vast.destroy_instance(id=instance_id)
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError("Vast did not confirm instance destruction.")
    finally:
        vast.client.timeout = previous_timeout
    print(f"Released instance: {instance_id}", flush=True)




