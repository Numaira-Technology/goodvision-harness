"""Offline budget-admission checks; no provider calls or private inputs."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.unified_benchmark.budget import BudgetGuard
from tools.unified_benchmark.contract import write_new, digest
from tools.unified_benchmark.retry import PausedRun


def intent(n):
    return dict(unit=str(n),attempt=1,request_sha256='a'*64)


def outcome(root,n,cost):
    metadata={'usage':{}} if cost is None else {'usage':{'cost':cost}}
    write_new(root/'phase/outcomes'/(digest(str(n))+'-attempt-1.json'),intent(n)|{'metadata':metadata})


def make(root,wait=1):
    return BudgetGuard(root,{'budget_usd':1,'budget_wait_seconds':wait},'a/validation',root/'phase',.6)


def test_wait_for_live_receipt_releases_reservation(tmp_path):
    g=make(tmp_path);g(intent(1))
    with ThreadPoolExecutor() as pool:
        f=pool.submit(g,intent(2));time.sleep(.08)
        assert not f.done()
        assert g.lock.acquire(timeout=.1)
        g.lock.release()
        outcome(tmp_path,1,.1);f.result(timeout=2)
    assert len(g.state['reservations'])==2
    assert g.state['charged_or_reserved_usd']==pytest.approx(.7)
    assert any(r.get('budget_admission_wait_seconds',0)>0 for r in g.state['reservations'].values())


def test_old_orphan_is_not_a_live_wait_target(tmp_path):
    make(tmp_path)(intent(1));g=make(tmp_path);start=time.monotonic()
    with pytest.raises(PausedRun,match='budget cap'):g(intent(2))
    assert time.monotonic()-start<.5
    assert len(g.state['reservations'])==1


def test_unknown_completed_charge_keeps_full_reservation(tmp_path):
    g=make(tmp_path);g(intent(1));outcome(tmp_path,1,None)
    with pytest.raises(PausedRun,match='budget cap'):g(intent(2))
    assert g.state['charged_or_reserved_usd']==.6


@pytest.mark.parametrize('reason',['stop','cancel'])
def test_stop_during_wait_cannot_admit_new_intent(tmp_path,reason):
    g=make(tmp_path);event=threading.Event();g.set_cancel_event(event);g(intent(1))
    with ThreadPoolExecutor() as pool:
        f=pool.submit(g,intent(2));time.sleep(.08)
        if reason=='stop':(tmp_path/'STOP').write_text('stop')
        else:event.set()
        with pytest.raises(PausedRun):f.result(timeout=2)
    assert len(g.state['reservations'])==1


def test_live_receipt_wait_has_deadline(tmp_path):
    g=make(tmp_path,wait=.1);g(intent(1));start=time.monotonic()
    with pytest.raises(PausedRun,match='budget cap'):g(intent(2))
    assert .09<=time.monotonic()-start<1


def test_receipt_arriving_after_reconcile_does_not_permanently_block(tmp_path,monkeypatch):
    from pathlib import Path
    g=make(tmp_path);g(intent(1))
    target=tmp_path/'phase/outcomes'/(digest('1')+'-attempt-1.json')
    original=Path.exists
    published=False
    def exists(path):
        nonlocal published
        if path==target and not published:
            published=True
            outcome(tmp_path,1,.1)
            return False
        return original(path)
    monkeypatch.setattr(Path,'exists',exists)
    g(intent(2))
    assert g.state['charged_or_reserved_usd']==pytest.approx(.7)
    assert len(g.state['reservations'])==2

