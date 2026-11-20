"""Offline v2 reference-overlay regressions; no private data or model calls."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.unified_benchmark import dataset202, preflight, supplement, workflow
from tools.unified_benchmark.contract import CURRENT_VERSION, digest, file_hash, score


def baseline_parts():
    """Synthetic original202/validation24 identities, with 192 test videos."""
    test = []
    for event, count in zip(dataset202.EVENTS, (32, 22, 62, 66, 10, 10)):
        for index in range(count):
            name = f"bag-{index}.mp4" if event.startswith("bag_") else f"{event}-{index}.mp4"
            number = len(test) + 1
            test.append({
                "number": number, "filename": name, "sha256": digest(name),
                "event": event, "source_video_group": name,
                "proposed_answer": "yes" if index % 2 else "no",
                "human_review": "user_reviewed" if number <= 120 else "pending_view_review",
                "reference_review": "inherited_from_human_reviewed_source",
            })
    validation = [
        {"number": index + 1, "filename": f"val-{index}.mp4",
         "sha256": digest(f"val-{index}"), "source_video_group": f"val-{index}",
         "event": dataset202.EVENTS[index // 6],
         "proposed_answer": "yes" if index % 6 < 3 else "no", "human_review": "user_reviewed"}
        for index in range(24)
    ]
    return {
        "validation": {"rows": validation, "split_context": {}},
        "test": {"rows": test, "split_context": {}, "dataset_context": {
            "dataset": "synthetic-original202", "test_questions": 202, "test_videos": 192,
            "review_status_by_cohort": deepcopy(dataset202.REVIEW_STATUS),
        }},
    }


def supplement_document(parts, *, reviewed):
    doc = {"schema": "event-test-supplement-v2" if reviewed else supplement.SCHEMA,
           "split": "test", "dataset": "synthetic-reviewed222", "samples": [
               {"number": 203 + 2 * index + offset, "filename": f"extra-{index}.mp4",
                "sha256": digest(f"extra-{index}"), "source_video_group": f"extra-{index}",
                "event": event, "proposed_answer": "yes" if index % 2 else "no",
                "human_review": "synthetic assistant annotation; unreviewed"}
               for index in range(10)
               for offset, event in enumerate(dataset202.EVENTS[4:])
           ]}
    if reviewed:
        # Include original120 and robot82, unchanged reviewed answers and boundary IDs.
        changed = {87, 167, 168, 193, 194}
        doc["reviewed_answers"] = []
        for number in (1, 87, 121, 167, 168, 193, 194, 202):
            row = parts["test"]["rows"][number - 1]
            answer = row["proposed_answer"]
            doc["reviewed_answers"].append({
                **{key: row[key] for key in ("number", "filename", "event", "sha256", "source_video_group")},
                "previous_answer": answer,
                "proposed_answer": ("no" if answer == "yes" else "yes") if number in changed else answer,
            })
    return doc


def validation_document():
    return {"schema": "bag-validation10-v1", "videos": [
        {"filename": f"validation-bag-{index}.mp4", "sha256": digest(f"validation-bag-{index}"),
         "source_video_group": f"validation-bag-{index}", "questions": [
             {"event": event, "proposed_answer": "yes" if index % 2 else "no", "human_review": "synthetic"}
             for event in dataset202.EVENTS[4:]
         ]}
        for index in range(10)
    ]}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_video(split, row, out, _media=None, _protocol=None):
    frames = []
    for index in range(2):
        relative = f"frames/{split}/{row['sha256']}/{index}.jpg"
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"offline synthetic frame")
        frames.append({"path": relative, "sha256": file_hash(path), "time": float(index), "size": [720, 480]})
    return {"filename": row["filename"], "frames": frames, "full_decode_passed": True}


class ReviewedSupplementTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")))
        self.stack.enter_context(patch("socket.create_connection", side_effect=AssertionError("network forbidden")))

    def apply(self, name, parts, document, *, bags=False):
        out = self.root / name
        (out / "inputs").mkdir(parents=True)
        if bags:
            validation = self.root / f"{name}-validation.json"
            write_json(validation, validation_document())
            dataset202.add_bag_validation(out, parts, validation, file_hash(validation))
        path = self.root / f"{name}-supplement.json"
        write_json(path, document)
        dataset202.add_test_supplement(out, parts, path)
        return out

    def test_changed_and_unchanged_reviews_preserve_requests_validation_and_denominator(self):
        for bags in (False, True):
            with self.subTest(validation_questions=44 if bags else 24):
                baseline = baseline_parts()
                original_rows = baseline["test"]["rows"]
                original_snapshot = deepcopy(original_rows)
                reviewed = deepcopy(baseline)
                reviewed_original_rows = reviewed["test"]["rows"]
                v1 = supplement_document(baseline, reviewed=False)
                v2 = supplement_document(reviewed, reviewed=True)
                v2["reviewed_answers"].reverse()
                self.apply(f"v1-{bags}", baseline, v1, bags=bags)
                out = self.apply(f"v2-{bags}", reviewed, v2, bags=bags)
                self.assertEqual(original_rows, original_snapshot)
                self.assertEqual(reviewed_original_rows, original_snapshot)
                self.assertEqual(reviewed["test"]["rows"][202:], baseline["test"]["rows"][202:])
                self.assertEqual(reviewed["test"]["dataset_context"]["test_supplement"]["reviewed_question_numbers"],
                                 sorted(row["number"] for row in v2["reviewed_answers"]))
                self.assertEqual(baseline["validation"], reviewed["validation"])
                self.assertEqual(len(reviewed["test"]["rows"]), 222)
                changed = {87, 167, 168, 193, 194}
                self.assertEqual({a["number"] for a, b in zip(baseline["test"]["rows"], reviewed["test"]["rows"])
                                  if a["proposed_answer"] != b["proposed_answer"]}, changed)
                for row in v2["reviewed_answers"]:
                    actual = reviewed["test"]["rows"][row["number"] - 1]
                    self.assertEqual(actual["human_review"], "user_reviewed")
                    self.assertEqual(actual["reference_review"], "user_reviewed")
                videos = {split: {r["filename"]: synthetic_video(split, r, out) for r in part["rows"]}
                          for split, part in reviewed.items()}
                aliases = [path.stem for path in preflight.PROFILES.glob("*.json")]
                self.assertTrue(aliases)
                for alias, config in preflight.model_configs(aliases).items():
                    with self.subTest(alias=alias):
                        plans = []
                        for parts in (baseline, reviewed):
                            plans.append({split: preflight.build_plan(split, part["rows"], videos[split], config, out,
                                          CURRENT_VERSION, deepcopy(part.get("dataset_context")))
                                          for split, part in parts.items()})
                        before, after = plans
                        self.assertEqual(before["validation"], after["validation"])
                        self.assertEqual(before["test"]["units"], after["test"]["units"])
                        self.assertEqual(before["test"]["rules_sha256"], after["test"]["rules_sha256"])
                        for old, new in zip(before["test"]["questions"], after["test"]["questions"]):
                            self.assertEqual({k: v for k, v in old.items() if k != "reference"},
                                             {k: v for k, v in new.items() if k != "reference"})
                        evidence = {}
                        for question in before["test"]["questions"]:
                            for uid in question["unit_ids"]:
                                evidence.setdefault(uid, {})[question["event"]] = {"answer": question["reference"]}
                        old_score = score(before["test"], evidence, "choice")
                        new_score = score(after["test"], evidence, "choice")
                        self.assertEqual((old_score["correct"], new_score["correct"]), (222, 217))
                        self.assertEqual(new_score["accuracy_denominator"], 222)
                        self.assertEqual({k: v["accuracy_denominator"] for k, v in new_score["cohorts"].items()},
                                         {"original120": 120, "robot82": 82, "test_supplement": 20})
                        supplement.verify_supplement_plan(after["test"], v2["samples"] + v2["reviewed_answers"],
                                                          {"dataset_context": reviewed["test"]["dataset_context"]})

    def test_invalid_review_rejected_before_save_or_mutation(self):
        mutations = {
            "duplicate": lambda doc: doc["reviewed_answers"].append(deepcopy(doc["reviewed_answers"][0])),
            "wrong_original_number": lambda doc: doc["reviewed_answers"][0].update(number=2),
            "zero": lambda doc: doc["reviewed_answers"][0].update(number=0),
            "supplement_number": lambda doc: doc["reviewed_answers"][0].update(number=203),
            "boolean_number": lambda doc: doc["reviewed_answers"][0].update(number=True),
            "float_number": lambda doc: doc["reviewed_answers"][0].update(number=1.0),
            "previous_answer": lambda doc: doc["reviewed_answers"][0].update(previous_answer="yes"),
            "bad_reference": lambda doc: doc["reviewed_answers"][0].update(proposed_answer="unknown"),
            "filename": lambda doc: doc["reviewed_answers"][0].update(filename="different.mp4"),
            "event": lambda doc: doc["reviewed_answers"][0].update(event="smoking"),
            "hash": lambda doc: doc["reviewed_answers"][0].update(sha256=digest("different")),
            "group": lambda doc: doc["reviewed_answers"][0].update(source_video_group="different"),
            "v1_with_reviews": lambda doc: doc.update(schema=supplement.SCHEMA),
            "v2_without_reviews": lambda doc: doc.pop("reviewed_answers"),
            "v2_empty_reviews": lambda doc: doc.update(reviewed_answers=[]),
            "not_a_list": lambda doc: doc.update(reviewed_answers={}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                parts = baseline_parts()
                before = deepcopy(parts)
                doc = supplement_document(parts, reviewed=True)
                mutate(doc)
                with self.assertRaises(ValueError):
                    self.apply(name, parts, doc)
                self.assertEqual(parts, before)
                self.assertFalse((self.root / name / "inputs/test-supplement.json").exists())

    def test_bundle_verification_covers_reviewed_reference_and_source_identity(self):
        sources = {"tools/unified_benchmark/" + path.name: path for path in (
            Path(dataset202.__file__), Path(supplement.__file__), Path(preflight.__file__), Path(workflow.__file__))}
        self.stack.enter_context(patch.object(preflight, "source_files", return_value=sources))
        self.stack.enter_context(patch.object(workflow, "source_files", return_value=sources))
        def load(out, *_args):
            (out / "inputs").mkdir()
            return baseline_parts()
        self.stack.enter_context(patch.object(preflight, "load_inputs", side_effect=load))
        self.stack.enter_context(patch.object(preflight, "inspect_video", side_effect=synthetic_video))
        self.stack.enter_context(patch.object(preflight.subprocess, "check_output", return_value="offline ffmpeg\n"))
        doc = supplement_document(baseline_parts(), reviewed=True)
        path = self.root / "input.json"
        write_json(path, doc)
        bundle = self.root / "bundle"
        with redirect_stdout(StringIO()):
            self.assertEqual(preflight.main(["--inputs", str(self.root), "--media", str(self.root),
                "--output", str(bundle), "--models", "clip", "qwen8b", "videojev",
                "--test-supplement", str(path)]), 0)
        workflow.verify_bundle(bundle)
        plan_path = bundle / "plans/clip-test.json"
        original = json.loads(plan_path.read_text())
        for field in ("reference", "filename", "event", "source_video_group", "source_id", "missing", "duplicate"):
            with self.subTest(field=field):
                plan = deepcopy(original)
                question = plan["questions"][166]  # Reviewed question167.
                if field == "source_id":
                    unit = next(unit for unit in plan["units"] if unit["id"] == question["unit_ids"][0])
                    unit["source_id"] = "test/" + digest("different")
                elif field == "missing":
                    plan["questions"].remove(question)
                elif field == "duplicate":
                    plan["questions"].append(deepcopy(question))
                elif field == "reference":
                    question["reference"] = next(row["previous_answer"] for row in doc["reviewed_answers"]
                                                 if row["number"] == question["number"])
                else:
                    question[field] = "different"
                write_json(plan_path, plan)
                report_path = bundle / "preflight.json"
                report = json.loads(report_path.read_text())
                report["models"]["clip"]["test"]["plan_sha256"] = digest(plan)
                write_json(report_path, report)
                hashes = {p.relative_to(bundle).as_posix(): file_hash(p) for p in bundle.rglob("*")
                          if p.is_file() and p.name != "bundle-files-sha256.json"}
                write_json(bundle / "bundle-files-sha256.json", hashes)
                with self.assertRaises(ValueError):
                    workflow.verify_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
