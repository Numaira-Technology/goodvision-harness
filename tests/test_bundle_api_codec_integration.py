"""Both historical receipt entry points retain strict outer-envelope validation."""
import json
import math

import pytest

from tools.unified_benchmark.contract import read_receipt, write_new, write_receipt
from tools.unified_benchmark.receipts import raw_fields


@pytest.mark.parametrize("codec", ["global", "field"])
def test_contract_reader_accepts_both_raw_codecs_without_rewriting(tmp_path, codec):
    path = tmp_path / "receipt.json"
    raw = {"score": .8, "optional": float("nan")}
    if codec == "global":
        write_receipt(path, {"response": raw})
    else:
        write_new(path, raw_fields(response=raw))
    before = path.read_bytes()
    receipt = read_receipt(path)
    assert receipt["response"]["score"] == .8
    assert math.isnan(receipt["response"]["optional"])
    assert path.read_bytes() == before
    json.loads(before, parse_constant=lambda token: pytest.fail(token))


@pytest.mark.parametrize("raw", [
    '{"response": NaN}',
    '{"response": 1e400}',
    '{"response": {}, "response": {}}',
    '{"response_json": "{}"}',
    '{"response": "{}", "response_encoding": "unknown"}',
    '{"raw_json_encoding":"python-json-nonfinite-v1", "response":{}, "response_json":"{}"}',
    '{"raw_json_encoding":"python-json-nonfinite-v1", "response":null, "response_json":"{}", "response_encoding":"json-text-nonfinite-v1"}',
])
def test_contract_reader_does_not_bypass_strict_receipt_guards(tmp_path, raw):
    path = tmp_path / "receipt.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_receipt(path)
    assert path.read_text() == raw
