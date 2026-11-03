"""Publish independent delivery files to R2 and fetch verified releases.

Only the publishing machine needs boto3 and R2 credentials. A test server uses
the standard-library fetch command with a short-lived download plan.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SUPPORT = (
    "handoff.json",
    "README.txt",
    "README-服务器安装.txt",
    "Install-Server.cmd",
    "Start-Server.cmd",
    "Status-Server.cmd",
    "Update-Module.cmd",
    "Rollback-Module.cmd",
)
PART_SIZE = 32 * 1024**2


def relative(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("Expected a relative delivery path")
    return value


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        os.chmod(temporary, 0o600 if private else 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def inventory(root):
    root = Path(root).resolve()
    handoff = json.loads((root / "handoff.json").read_text())
    entries = {}
    for entry in handoff["payloads"]:
        name = relative(entry["file"])
        if name in entries:
            raise ValueError("Duplicate delivery path")
        entries[name] = entry
    for name in SUPPORT:
        if (root / name).is_file():
            entries.setdefault(name, {"file": name})
    for name in list(entries):
        if name.endswith(".tar.gz"):
            metadata = name.removesuffix(".tar.gz") + ".json"
            if (root / metadata).is_file():
                entries.setdefault(metadata, {"file": metadata})
    result = []
    for name, expected in sorted(entries.items()):
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Delivery file is missing or escapes its root")
        size, checksum = path.stat().st_size, digest(path)
        if ("sha256" in expected and expected["sha256"] != checksum) or (
            "bytes" in expected and expected["bytes"] != size
        ):
            raise ValueError("Delivery file differs from handoff.json: " + name)
        result.append({"file": name, "bytes": size, "sha256": checksum})
    return {
        "schema": 1,
        "delivery_version": handoff["delivery_version"],
        "installer_revision": handoff["installer_revision"],
        "module_versions": handoff["module_versions"],
        "files": result,
    }


def validate_manifest(value, *, urls=False):
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("Unsupported release manifest")
    entries = value.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Release has no files")
    names = set()
    for entry in entries:
        name = relative(entry["file"])
        relative(entry["key"])
        if name in names:
            raise ValueError("Duplicate release file")
        names.add(name)
        if (
            type(entry["bytes"]) is not int
            or entry["bytes"] < 0
            or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
        ):
            raise ValueError("Invalid release file identity")
        if urls:
            parsed = urlsplit(entry["url"])
            if parsed.scheme != "https" or not parsed.hostname or parsed.username:
                raise ValueError("Expected an HTTPS download URL")
    # Reject file/directory collisions before writing any download.
    if any(
        str(parent) in names
        for name in names
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    ):
        raise ValueError("Release file collides with a directory")
    if any(name + ".r2-partial" in names for name in names):
        raise ValueError("Release file collides with a partial download")
    return value


def client(env_file):
    import boto3
    from botocore.config import Config

    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    names = (
        "R2_ENDPOINT_URL",
        "R2_BUCKET",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
    )
    if any(not os.environ.get(name) for name in names):
        raise ValueError("Configure " + ", ".join(names))
    endpoint = urlsplit(os.environ["R2_ENDPOINT_URL"])
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username:
        raise ValueError("R2 endpoint must use HTTPS")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(
            signature_version="s3v4",
            connect_timeout=20,
            read_timeout=90,
            retries={"max_attempts": 4},
            max_pool_connections=12,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    ), os.environ["R2_BUCKET"]


def head(s3, bucket, key):
    from botocore.exceptions import ClientError

    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return None
        raise


def upload_file(s3, bucket, root, entry, state, checkpoint, workers):
    key, size = entry["key"], entry["bytes"]
    found = head(s3, bucket, key)
    if found:
        if (
            found["ContentLength"] != size
            or found.get("Metadata", {}).get("sha256") != entry["sha256"]
        ):
            raise ValueError("Existing object identity differs; no overwrite: " + key)
        return False
    item = state.get(key)
    if item and (item["sha256"] != entry["sha256"] or item["bytes"] != size):
        raise ValueError("Multipart checkpoint identity differs")
    if size < 5 * 1024**2:
        payload = (root / entry["file"]).read_bytes()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            Metadata={"sha256": entry["sha256"]},
            IfNoneMatch="*",
        )
        return True
    if not item:
        response = s3.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            Metadata={"sha256": entry["sha256"]},
        )
        item = {
            "id": response["UploadId"],
            "bytes": size,
            "sha256": entry["sha256"],
            "part_size": PART_SIZE,
        }
        state[key] = item
        write_json(checkpoint, state, private=True)
    upload_id, part_size = item["id"], item["part_size"]
    existing = {
        part["PartNumber"]: part
        for page in s3.get_paginator("list_parts").paginate(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
        )
        for part in page.get("Parts", [])
    }

    def send(number):
        offset = (number - 1) * part_size
        with (root / entry["file"]).open("rb") as source:
            source.seek(offset)
            data = source.read(min(part_size, size - offset))
        checksum = hashlib.md5(data).digest()
        previous = existing.get(number)
        if (
            previous
            and previous["Size"] == len(data)
            and previous["ETag"].strip('"') == checksum.hex()
        ):
            etag = previous["ETag"]
        else:
            etag = s3.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=number,
                Body=data,
                ContentMD5=base64.b64encode(checksum).decode(),
            )["ETag"]
            if etag.strip('"') != checksum.hex():
                raise ValueError("Uploaded part checksum differs")
        return {"PartNumber": number, "ETag": etag}

    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(send, n) for n in range(1, math.ceil(size / part_size) + 1)
        ]
        for future in as_completed(futures):
            parts.append(future.result())
    if digest(root / entry["file"]) != entry["sha256"]:
        raise ValueError("Local file changed during upload")
    s3.complete_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": sorted(parts, key=lambda part: part["PartNumber"])},
    )
    found = head(s3, bucket, key)
    if (
        found["ContentLength"] != size
        or found["Metadata"].get("sha256") != entry["sha256"]
    ):
        raise ValueError("Completed object identity differs")
    state.pop(key)
    write_json(checkpoint, state, private=True)
    return True


def publish(s3, bucket, root, release, checkpoint, previous=None, workers=8):
    relative(release)
    value = inventory(root)
    old = validate_manifest(previous)["files"] if previous else []
    reusable = {(entry["sha256"], entry["bytes"]): entry["key"] for entry in old}
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    # Content keys keep baselines and weights reusable across releases.
    for entry in value["files"]:
        entry["key"] = reusable.get(
            (entry["sha256"], entry["bytes"]),
            "artifacts/sha256/" + entry["sha256"],
        )
    key = "releases/" + release + "/r2-release-manifest.json"
    body = json.dumps(value, indent=2, ensure_ascii=False).encode()
    found = head(s3, bucket, key)
    if found:
        if s3.get_object(Bucket=bucket, Key=key)["Body"].read() != body:
            raise ValueError("Release already exists with different content")
        for entry in value["files"]:
            require_object(s3, bucket, entry)
        return key
    for entry in value["files"]:
        if entry["key"] in reusable.values():
            require_object(s3, bucket, entry)
            continue
        changed = upload_file(s3, bucket, Path(root), entry, state, checkpoint, workers)
        print(("Uploaded: " if changed else "Reused: ") + entry["file"], flush=True)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        IfNoneMatch="*",
    )
    return key


def require_object(s3, bucket, entry):
    found = head(s3, bucket, entry["key"])
    if (
        not found
        or found["ContentLength"] != entry["bytes"]
        or found.get("Metadata", {}).get("sha256", entry["sha256"]) != entry["sha256"]
    ):
        raise ValueError("Release object is missing or differs: " + entry["file"])


def download_plan(s3, bucket, key, seconds=3600):
    value = validate_manifest(
        json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    )
    for entry in value["files"]:
        require_object(s3, bucket, entry)
    for entry in value["files"]:
        entry["url"] = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": entry["key"]},
            ExpiresIn=seconds,
        )
    value["expires_at"] = time.time() + seconds
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(plan, destination, workers=4):
    validate_manifest(plan, urls=True)
    if plan.get("expires_at", 0) <= time.time():
        raise ValueError("Download plan expired; generate new URLs")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Check every destination before the first write, including existing links.
    for entry in plan["files"]:
        path = destination / entry["file"]
        if not path.resolve().is_relative_to(destination):
            raise ValueError("Download destination escapes the delivery directory")

    def receive(entry):
        path = destination / entry["file"]
        if path.exists():
            if (
                path.stat().st_size == entry["bytes"]
                and digest(path) == entry["sha256"]
            ):
                return
            raise ValueError("Existing delivery file differs: " + entry["file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".r2-partial")
        if partial.is_symlink():
            raise ValueError("Partial download must not be a symbolic link")
        if entry["bytes"] == 0:
            partial.touch(exist_ok=True)
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry["bytes"]:
                raise ValueError("Partial download exceeds the declared size")
            if offset == entry["bytes"]:
                break
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with build_opener(NoRedirect).open(
                    Request(entry["url"], headers=headers),
                    timeout=60,
                ) as source:
                    append = offset and source.status == 206
                    received = offset if append else 0
                    with partial.open("ab" if append else "wb") as output:
                        while chunk := source.read(4 * 1024**2):
                            received += len(chunk)
                            if received > entry["bytes"]:
                                raise ValueError("Download exceeds the declared size")
                            output.write(chunk)
                if received == entry["bytes"]:
                    break
            except (OSError, HTTPError):
                if attempt == 2:
                    raise RuntimeError("Download failed: " + entry["file"]) from None
                time.sleep(attempt + 1)
        if (
            partial.stat().st_size != entry["bytes"]
            or digest(partial) != entry["sha256"]
        ):
            partial.unlink(missing_ok=True)
            raise ValueError("Download checksum differs: " + entry["file"])
        os.chmod(partial, 0o644)
        os.replace(partial, path)
        print("Verified: " + entry["file"], flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(receive, plan["files"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    upload = sub.add_parser(
        "publish", help="Upload changed files, then publish an immutable release"
    )
    upload.add_argument("root", type=Path)
    upload.add_argument("--release", required=True)
    upload.add_argument("--previous", help="Previous R2 release manifest key")
    upload.add_argument("--checkpoint", type=Path, required=True)
    upload.add_argument("--env-file", type=Path)
    urls = sub.add_parser("urls", help="Write private temporary URLs for a test server")
    urls.add_argument("manifest_key")
    urls.add_argument("--output", type=Path, required=True)
    urls.add_argument("--seconds", type=int, default=3600)
    urls.add_argument("--env-file", type=Path)
    download = sub.add_parser(
        "fetch", help="Download and verify using only the standard library"
    )
    download.add_argument("plan", type=Path)
    download.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "fetch":
            fetch(json.loads(args.plan.read_text()), args.directory)
            return
        s3, bucket = client(args.env_file)
        if args.action == "publish":
            previous = (
                json.loads(
                    s3.get_object(Bucket=bucket, Key=args.previous)["Body"].read()
                )
                if args.previous
                else None
            )
            print(
                publish(s3, bucket, args.root, args.release, args.checkpoint, previous)
            )
        else:
            if not 1 <= args.seconds <= 7 * 24 * 60 * 60:
                raise ValueError(
                    "URL lifetime must be between one second and seven days"
                )
            write_json(
                args.output,
                download_plan(s3, bucket, args.manifest_key, args.seconds),
                private=True,
            )
            print("Temporary download plan saved: " + str(args.output))
    except Exception as error:
        # SDK/HTTP exception strings can contain credentials or signed URLs.
        print("R2 delivery failed (" + type(error).__name__ + ")")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
