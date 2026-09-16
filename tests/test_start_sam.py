from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "start_sam", Path(__file__).resolve().parents[1] / "tools/start_sam.py",
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class WorkerBundleTests(unittest.TestCase):
    def test_upload_contains_all_local_import_dependencies(self):
        worker = launcher.ROOT / "workers" / "sam_scene"
        local_modules = {path.stem for path in worker.glob("*.py")}
        bundled = {Path(name).stem for name in launcher.WORKER_FILES}
        for name in launcher.WORKER_FILES:
            source = worker / name
            self.assertTrue(source.is_file(), f"Missing packaged worker file: {name}")
            if source.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    dependencies = [node.module.split(".")[0]] if node.module else [alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    dependencies = [alias.name.split(".")[0] for alias in node.names]
                else:
                    continue
                for dependency in dependencies:
                    if dependency in local_modules:
                        self.assertIn(dependency, bundled, f"{name} imports missing {dependency}")


class WorkerReadinessTests(unittest.TestCase):
    def test_waits_for_both_routes_including_hybrid_enablement(self):
        opener = Mock()
        opener.open.side_effect = [
            io.BytesIO(json.dumps({"enabled": True, "ready": True, "gpu_ready": True}).encode()),
            io.BytesIO(json.dumps({"enabled": False, "ready": True, "gpu_ready": True}).encode()),
            io.BytesIO(json.dumps({"enabled": True, "ready": True, "gpu_ready": True}).encode()),
            io.BytesIO(json.dumps({"enabled": True, "ready": True, "gpu_ready": True}).encode()),
        ]
        process = Mock()
        process.poll.return_value = None
        with patch.object(launcher, "build_opener", return_value=opener), patch.object(launcher.time, "sleep"):
            launcher.wait_for_worker(process, 8092)
        self.assertEqual(
            [call.args[0] for call in opener.open.call_args_list],
            ["http://127.0.0.1:8092/health/perception/" + route
             for route in ("sam_live", "hybrid", "sam_live", "hybrid")],
        )


@unittest.skipUnless(os.name == "posix" and shutil.which("ssh-keygen"), "Linux SSH permissions")
class SSHKeyTests(unittest.TestCase):
    def test_windows_mount_permissions_are_fixed_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "host_key"
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(source)],
                check=True, capture_output=True,
            )
            source.chmod(0o777)
            private = root / "private"
            private.mkdir(mode=0o700)
            copied = launcher.prepare_ssh_key(source, private)
            self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o777)
            self.assertEqual(copied.read_bytes(), source.read_bytes())

    def test_encrypted_key_is_rejected_without_a_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "encrypted"
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "test-passphrase", "-f", str(source)],
                check=True, capture_output=True,
            )
            with self.assertRaisesRegex(RuntimeError, "before renting"):
                launcher.prepare_ssh_key(source, root)

    def test_invalid_key_stops_before_any_vast_request(self):
        import vastai

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid"
            source.write_text("invalid private key")
            with (
                patch.dict(os.environ, {"VAST_API_KEY": "unused", "HF_TOKEN": "unused", "SAM_SSH_KEY": str(source)}),
                patch("sys.argv", ["start_sam.py"]),
                patch.object(vastai, "VastAI") as client,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(launcher.main(), 1)
                client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
