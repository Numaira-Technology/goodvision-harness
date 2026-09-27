"""Keep the original Windows launcher compatible with the updated image."""

import base64
import hashlib
import json

NAMES = ("RELEASE.json", "SOURCE_HASHES.json", "FILES.json")


def read_metadata(root, base_id):
    if root is None:
        return {}
    files = {}
    for name in NAMES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing original Windows release metadata: {name}")
        files[name] = path.read_bytes()
    release = json.loads(files["RELEASE.json"].decode("utf-8-sig"))
    if base_id not in release.get("accepted_image_ids", []):
        raise RuntimeError("Windows launcher metadata does not match installed image.")
    inventory = json.loads(files["FILES.json"].decode("utf-8-sig"))
    for name in NAMES[:2]:
        item = next((f for f in inventory["files"] if f["path"] == name), None)
        if item is None or hashlib.sha256(files[name]).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Original release metadata checksum mismatch: {name}")
    return {name: base64.b64encode(data).decode() for name, data in files.items()}


def atomic_write(path, data):
    temporary = path.with_name(path.name + ".goodvision-update.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def update_metadata(root, backup, manifest, image_id):
    if not backup:
        return
    if root is None:
        raise RuntimeError("The original installation folder is required.")
    original = {
        name: json.loads(base64.b64decode(value).decode("utf-8-sig"))
        for name, value in backup.items()
    }
    release = original["RELEASE.json"]
    release["accepted_image_ids"] = sorted(
        set(release["accepted_image_ids"] + [image_id])
    )
    release["ui_update"] = {"release": manifest["release"], "image_id": image_id}
    source = original["SOURCE_HASHES.json"]
    source["image_id"] = image_id
    source["source"] = "Installed source after " + manifest["release"]
    for item in manifest["files"]:
        if item["path"].startswith("payload/"):
            source["files"][item["path"][8:]] = item["sha256"]
    encoded = {
        name: (json.dumps(original[name], indent=2) + "\n").encode()
        for name in NAMES[:2]
    }
    for item in original["FILES.json"]["files"]:
        if item["path"] in encoded:
            data = encoded[item["path"]]
            item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    encoded["FILES.json"] = (
        json.dumps(original["FILES.json"], indent=2) + "\n"
    ).encode()
    for name in NAMES:
        atomic_write(root / name, encoded[name])
