"""Deterministic module partition and incremental evidence for Linux CI."""

from __future__ import annotations

import faulthandler
import json
import os
import threading
import time
from pathlib import Path

import pytest

from tools.ci_regression import shard_for
from tools.synthetic_evidence import retain_synthetic_evidence

# GitHub cancels a job that reaches its timeout-minutes and keeps no log for
# it, so a hung Windows shard leaves nothing to diagnose. On Windows CI only,
# end the shard first: dump every thread and exit nonzero, keeping the log.
_WATCHDOG_ARMED = "GOODVISION_CI_WATCHDOG_ARMED"
_WATCHDOG_SECONDS = 20 * 60


def _arm_windows_watchdog(config):
    if os.name != "nt" or os.environ.get("GITHUB_ACTIONS") != "true":
        return
    if os.environ.get(_WATCHDOG_ARMED):
        return  # nested in-process or child pytest sessions
    os.environ[_WATCHDOG_ARMED] = "1"
    # Same approach as pytest's faulthandler plugin: duplicate the real stderr
    # now, before test capture can redirect file descriptor 2.
    stream = os.fdopen(os.dup(2), "w", buffering=1)

    def expire():
        stream.write(
            f"\n[ci watchdog] shard exceeded {_WATCHDOG_SECONDS}s; last stage: "
            f"{getattr(config, '_ci_last_stage', None)}\n"
        )
        stream.flush()
        faulthandler.dump_traceback(file=stream, all_threads=True)
        stream.flush()
        os._exit(3)

    timer = threading.Timer(_WATCHDOG_SECONDS, expire)
    timer.daemon = True
    timer.start()


def pytest_addoption(parser):
    group = parser.getgroup("ci evidence")
    group.addoption("--ci-shard", type=int, required=True)
    group.addoption("--ci-shards", type=int, required=True)
    group.addoption("--ci-evidence", required=True)


def pytest_configure(config):
    index, count = config.option.ci_shard, config.option.ci_shards
    if not 0 <= index < count:
        raise pytest.UsageError("CI shard must satisfy 0 <= index < count")
    if config.option.keyword or config.option.markexpr or config.option.deselect:
        raise pytest.UsageError("CI collection cannot use -k, -m or --deselect")
    directory = Path(config.option.ci_evidence)
    directory.mkdir(parents=True, exist_ok=True)
    config._ci_progress = (directory / "progress.jsonl").open("a", buffering=1)
    _record(config, event="configured", shard=index, shards=count)
    _arm_windows_watchdog(config)


def _record(config, **value):
    if value.get("event") == "stage":
        config._ci_last_stage = f"{value['nodeid']} ({value['stage']})"
    value["time_unix"] = time.time()
    config._ci_progress.write(json.dumps(value, ensure_ascii=True) + "\n")


def pytest_collection_modifyitems(session, config, items):
    all_nodes = [item.nodeid for item in items]
    if len(set(all_nodes)) != len(all_nodes):
        raise pytest.UsageError("CI collection contains duplicate node IDs")
    index, count = config.option.ci_shard, config.option.ci_shards
    selected, excluded = [], []
    for item in items:
        (selected if shard_for(item.nodeid, count) == index else excluded).append(item)
    manifest = {
        "schema": 1,
        "shard": index,
        "shards": count,
        "collected": all_nodes,
        "selected": [item.nodeid for item in selected],
    }
    path = Path(config.option.ci_evidence) / "collection.json"
    path.write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    items[:] = selected
    config.hook.pytest_deselected(items=excluded)
    _record(config, event="collection", collected=len(all_nodes), selected=len(items))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    _record(item.config, event="stage", nodeid=item.nodeid, stage="setup")
    yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    _record(item.config, event="stage", nodeid=item.nodeid, stage="call")
    yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    _record(item.config, event="stage", nodeid=item.nodeid, stage="teardown")
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    value = dict(
        event="result",
        nodeid=item.nodeid,
        stage=report.when,
        outcome=report.outcome,
        seconds=report.duration,
    )
    if report.failed or report.skipped:
        value["detail"] = str(report.longrepr)
    _record(item.config, **value)
    if report.failed:
        try:
            retain_synthetic_evidence(item, report.when)
        except Exception as exc:
            # A missing artifact must never turn a failed test into a plugin
            # internal error, or obscure its original assertion/traceback.
            report.sections.append(
                ("synthetic evidence retention error", type(exc).__name__)
            )


def pytest_sessionfinish(session, exitstatus):
    _record(session.config, event="finished", exit_code=int(exitstatus))


def pytest_unconfigure(config):
    stream = getattr(config, "_ci_progress", None)
    if stream is not None:
        stream.close()
