"""Fetch the exact historical RF checkpoint generation and enforce its identity."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

root = Path(sys.argv[1])
seed = json.loads((root / 'seed-weights.json').read_text())['models']['rfdetr-l']
expected_sha = '86879a31a42ea4c091ee8d758654bf5c8c465f773f08e95a9143653f0f529cba'
expected_bytes = 1571684963
expected_generation = '1742411445584539'
expected_url = 'https://storage.googleapis.com/rfdetr/rf-detr-large.pth'
expected_etag = '"992c8e862aa733a7bb2777e45d49f1a0"'
if (seed['sha256'] != expected_sha or seed['bytes'] != expected_bytes
        or seed['url'] != expected_url
        or seed['response_metadata']['x-goog-generation'] != expected_generation
        or seed['response_metadata']['ETag'] != expected_etag):
    raise RuntimeError('GLOBAL_RF_SEED_IDENTITY_MISMATCH')
target = root / 'rf-detr-large.pth'
partial = root / 'rf-detr-large.pth.part'
if target.exists() or partial.exists():
    raise RuntimeError('GLOBAL_RF_FETCH_ALREADY_ATTEMPTED')
started = time.time()
start_clock = time.monotonic()
download_url = expected_url + '?generation=' + expected_generation
print(json.dumps({'event': 'pinned_weight_download_started', 'url': download_url,
                  'expected_bytes': expected_bytes, 'sha256': expected_sha}), flush=True)
digest = hashlib.sha256()
size = 0
last_log = start_clock
request = Request(download_url, headers={'If-Match': expected_etag})
with urlopen(request, timeout=60) as response, partial.open('xb') as output:
    metadata = {key: response.headers.get(key) for key in
                ('Content-Length', 'ETag', 'x-goog-generation', 'x-goog-hash')}
    if (response.status != 200 or metadata['ETag'] != expected_etag
            or metadata['x-goog-generation'] != expected_generation
            or int(metadata['Content-Length'] or -1) != expected_bytes):
        raise RuntimeError('GLOBAL_RF_DOWNLOAD_OBJECT_CHANGED')
    while block := response.read(1024 * 1024):
        size += len(block)
        if size > expected_bytes or time.monotonic() - start_clock > 1200:
            raise RuntimeError('GLOBAL_RF_DOWNLOAD_BOUND_EXCEEDED')
        output.write(block)
        digest.update(block)
        if time.monotonic() - last_log >= 15:
            print(json.dumps({'event': 'pinned_weight_download_progress', 'bytes': size,
                              'expected_bytes': expected_bytes}), flush=True)
            last_log = time.monotonic()
    output.flush()
    os.fsync(output.fileno())
if size != expected_bytes or digest.hexdigest() != expected_sha:
    raise RuntimeError('GLOBAL_RF_WEIGHT_HASH_MISMATCH')
os.replace(partial, target)
receipt = {'started_epoch': started, 'finished_epoch': time.time(), 'bytes': size,
           'sha256': digest.hexdigest(), 'response_metadata': metadata,
           'source': 'historical_official_generation', 'url': download_url,
           'same_bytes_as_preserved_local_checkpoint': True}
(root / 'weights.json').write_text(json.dumps({'recorded_epoch': time.time(),
    'models': {'rfdetr-l': seed}, 'acquisition': receipt}, indent=2) + '\n')
print(json.dumps({'event': 'pinned_weight_download_completed', **receipt}), flush=True)
