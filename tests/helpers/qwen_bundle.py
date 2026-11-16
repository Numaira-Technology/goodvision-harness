"""Synthetic sealed Qwen bundle; no inference, media download, or process locks."""
from tools.unified_benchmark import preflight, workflow
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import RULES, digest, file_hash, write_new


def frozen_host_bundle(root):
    root.mkdir()
    config = preflight.model_configs(["qwen8b_r2_host"])["qwen8b_r2_host"]
    write_new(root / "configs/qwen8b_r2_host.json", config)
    frame = root / "frame.jpg"
    frame.write_bytes(b"synthetic offline JPEG bytes")
    unit = {"id": "u1", "targets": ["fight"], "question": "Is fighting visible?",
            "frames": [{"path": frame.name, "sha256": file_hash(frame), "time": 0}]}
    unit["request_sha256"] = digest(request_body(config, unit, root)[1])
    models = {"qwen8b_r2_host": {}}
    for split in ("validation", "test"):
        plan = {"protocol": RULES["version"], "split": split, "config_sha256": digest(config),
                "rules_sha256": digest(RULES), "units": [unit], "questions": [
                    {"number": 1, "event": "fight", "reference": "yes", "unit_ids": [unit["id"]]}]}
        write_new(root / f"plans/qwen8b_r2_host-{split}.json", plan)
        models["qwen8b_r2_host"][split] = {"plan_sha256": digest(plan)}
    write_new(root / "rules.json", RULES)
    sources = preflight.freeze_sources(root)
    write_new(root / "preflight.json", {"offline_passed": True, "sources_sha256": digest(sources), "models": models})
    write_new(root / "bundle-files-sha256.json", {p.relative_to(root).as_posix(): file_hash(p)
                                                 for p in root.rglob('*') if p.is_file()})
    workflow.verify_bundle(root)
    return config, unit
