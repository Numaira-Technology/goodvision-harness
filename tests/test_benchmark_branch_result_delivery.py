"""Historical benchmark results must not overwrite live results."""

from test_branch_result_delivery import observation

from tools.benchmark_runtime.result_bus import ResultBus


def test_historical_reply_is_delivered_but_never_populates_live_feature_cache():
    bus = ResultBus()
    old = {**observation("smoking", 1.0).as_dict(), "historical_only": True}
    bus.publish(old)
    assert bus.latest() is None
    assert bus.latest(feature="smoking") is None
    new = {**observation("smoking", 2.0).as_dict(), "judgments": []}
    bus.publish(new)
    bus.publish(old)
    assert bus.latest().payload == new
    assert bus.latest(feature="smoking") is None
    assert [r.payload for r in bus.records_after(0)] == [old, new, old]
