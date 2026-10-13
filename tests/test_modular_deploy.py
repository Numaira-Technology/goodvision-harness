"""Independent application/SAM/VLM plans and real Compose rendering, no GPU needed."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compose_support import isolated_docker_config

from tools import start_modular as launcher

ROOT = Path(__file__).resolve().parents[1]
DEVICES = [{"index": 3, "name": "NVIDIA L40S", "memory_mib": 46068}]


class ModularLauncherTests(unittest.TestCase):
    def test_server_origin_acceptance_matches_runtime_validation(self):
        from goodvision_harness.service_locations import validate_service_location

        origins = (
            "",
            "http://SAM:8092/",
            "  https://vlm.example  ",
            "http://[::1]:8092",
            "http://sam_host:8092",
            "http://sam:0",
            "http://sam:65536",
            "http://sam:",
            "http://sam?",
            "http://sam#",
            "http://sam%20host",
            "http://user:password@sam",
            "http://sam/path",
            "http://sam\\host",
            "http://sam\x00host",
            "http://sam\x7fhost",
            "http://bad..host",
            "http://-bad.host",
            "ftp://sam:8092",
            "http://" + "a" * 2050,
        )
        for origin in origins:
            with self.subTest(origin=repr(origin)):
                try:
                    expected = validate_service_location("tracking_server_url", origin)
                except ValueError:
                    with self.assertRaises(launcher.LaunchError):
                        launcher.server_origin(origin, "SAM server")
                else:
                    self.assertEqual(
                        launcher.server_origin(origin, "SAM server"), expected
                    )

    def test_main_model_alias_has_runtime_length_and_control_character_limits(self):
        for model in (" ", "a" * 513, "model\nname", "model\x7f"):
            args = launcher.parser().parse_args(["main", "--inference-model", model])
            with (
                self.subTest(model=repr(model)),
                self.assertRaises(launcher.LaunchError),
            ):
                launcher.main_environment({}, args)
        args = launcher.parser().parse_args(
            ["main", "--inference-model", "  Qwen3-VL  "]
        )
        self.assertEqual(
            launcher.main_environment({}, args)["INFERENCE_SERVER_MODEL"], "Qwen3-VL"
        )

    def test_main_starts_only_gateway_for_remote_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with (
                patch.object(launcher, "ROOT", Path(temporary)),
                patch.dict(
                    os.environ, {"INFERENCE_API_KEY": "cloud-secret"}, clear=True
                ),
                patch.object(launcher.local, "detect_nvidia_devices") as detect,
                patch.object(launcher.local, "validate_weights") as weights,
                patch.object(launcher, "validate_sam_weights") as sam,
                patch.object(launcher.subprocess, "run") as run,
                contextlib.redirect_stdout(output),
            ):
                run.return_value.returncode = 0
                code = launcher.main(
                    [
                        "main",
                        "--streams",
                        "16",
                        "--tracking-server",
                        "http://sam-host:8092",
                    ]
                )
            self.assertEqual(code, 0)
            detect.assert_not_called()
            weights.assert_not_called()
            sam.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertNotIn("cloud-secret", output.getvalue())
            self.assertEqual(plan["devices"], [])
            self.assertEqual(plan["environment"]["TRACKING_EXECUTION_MODE"], "server")
            self.assertEqual(plan["environment"]["INFERENCE_EXECUTION_MODE"], "cloud")
            self.assertEqual(plan["environment"]["TRACKING_SERVER_CAPACITY"], "16")
            self.assertEqual(plan["environment"]["LOCAL_GPU_AVAILABLE"], "false")
            self.assertEqual(
                run.call_args.kwargs["env"]["INFERENCE_API_KEY"], "cloud-secret"
            )
            self.assertEqual(plan["command"][-1], "harness")
            self.assertEqual(plan["environment"]["HARNESS_ANALYSIS_URL"], "http://sam-host:8080")
            self.assertEqual(plan["analysis_location"], "gpu_host")
            from goodvision_harness.whep_process import DEFAULT_FRAME_CAPACITY

            budget = plan["environment"]["HARNESS_GATEWAY_SHM_SIZE"]
            self.assertGreater(
                int(budget.removesuffix("m")) * 1024**2,
                2 * 16 * DEFAULT_FRAME_CAPACITY,
            )
            self.assertEqual(
                run.call_args.kwargs["env"]["HARNESS_GATEWAY_SHM_SIZE"], budget
            )
            self.assertIsNone(plan["fleet_overlay"])
            self.assertNotIn("sam-server", plan["command"])
            self.assertNotIn("vlm-server", plan["command"])
            self.assertFalse(
                (Path(temporary) / ".goodvision/main-server-compose.yaml").is_file()
            )

    def test_dry_run_does_not_write_or_call_processes(self):
        for argv in (
            ["main", "--streams", "1"],
            [
                "main",
                "--streams",
                "100",
                "--tracking-server",
                "http://sam:8092",
                "--inference-server",
                "https://vlm.example",
                "--inference-model",
                "qwen-vl",
            ],
            ["sam", "--streams", "16", "--gpu-count", "2", "--gpu-id", "1"],
            ["vlm", "--gpu-count", "1"],
        ):
            output = io.StringIO()
            with (
                self.subTest(argv=argv),
                patch.dict(os.environ, {}, clear=True),
                patch.object(launcher.local, "read_local_settings", return_value={}),
                patch.object(launcher.subprocess, "run") as run,
                patch.object(Path, "write_text") as write,
                contextlib.redirect_stdout(output),
            ):
                code = launcher.main([*argv, "--dry-run"])
            self.assertEqual(code, 0)
            run.assert_not_called()
            write.assert_not_called()
            plan = json.loads(output.getvalue())
            if argv[0] == "vlm":
                self.assertNotIn("SAM31_CHECKPOINT_FILE", plan["environment"])
                self.assertEqual(
                    plan["environment"]["LOCAL_INFERENCE_MAX_NUM_SEQS"], "8"
                )
                self.assertEqual(
                    plan["environment"]["LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION"],
                    "0.90",
                )
            if argv[0] == "sam":
                self.assertNotIn("LOCAL_INFERENCE_MODEL_DIRECTORY", plan["environment"])
                self.assertEqual(plan["environment"]["SAM_SERVER_GPU_ID"], "1")

    def test_sam_starts_with_only_sam_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "sam.pt"
            checkpoint.write_bytes(b"actual checkpoint fixture")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    launcher.local,
                    "read_local_settings",
                    return_value={
                        "SAM31_CHECKPOINT_FILE": str(checkpoint),
                    },
                ),
                patch.object(
                    launcher.local, "detect_nvidia_devices", return_value=DEVICES
                ),
                patch.object(launcher.local, "validate_weights") as vlm,
                patch.object(launcher.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run.return_value.returncode = 0
                code = launcher.main(["sam", "--streams", "12", "--gpu-id", "3"])
            self.assertEqual(code, 0)
            vlm.assert_not_called()
            self.assertEqual(run.call_args.args[0][-1], "sam-server")
            self.assertEqual(run.call_args.kwargs["env"]["SAM31_LIVE_CAMERAS"], "12")

    def test_vlm_starts_with_no_sam_checkpoint(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher.local, "read_local_settings", return_value={}),
            patch.object(launcher.local, "detect_nvidia_devices", return_value=DEVICES),
            patch.object(launcher.local, "validate_weights") as weights,
            patch.object(launcher, "validate_sam_weights") as sam,
            patch.object(launcher.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run.return_value.returncode = 0
            code = launcher.main(["vlm", "--gpu-id", "3"])
        self.assertEqual(code, 0)
        sam.assert_not_called()
        self.assertEqual(weights.call_args.kwargs, {"require_tracking": False})
        self.assertEqual(weights.call_args.args[0]["LOCAL_INFERENCE_GPU_ID"], "3")
        self.assertEqual(run.call_args.args[0][-2:], ["vlm-server", "vlm-processing"])
        self.assertIn("--build", run.call_args.args[0])

    def test_complete_vlm_weight_validation_does_not_touch_sam(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, content in {
                "config.json": {"model_type": "qwen3_vl_moe"},
                "preprocessor_config.json": {
                    "image_processor_type": "Qwen3VLImageProcessor"
                },
                "video_preprocessor_config.json": {
                    "video_processor_type": "Qwen3VLProcessor"
                },
                "tokenizer_config.json": {"chat_template": "{{ messages }}"},
                "tokenizer.json": {"version": "1.0"},
            }.items():
                (directory / name).write_text(json.dumps(content), encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"test weights")
            environment = {"LOCAL_INFERENCE_MODEL_DIRECTORY": temporary}
            launcher.local.validate_weights(environment, require_tracking=False)
            (directory / "model.safetensors").write_bytes(
                b"version https://git-lfs.github.com/spec/v1"
            )
            with self.assertRaises(launcher.LaunchError):
                launcher.local.validate_weights(environment, require_tracking=False)

    def test_invalid_origins_modes_capacity_or_devices_fail_before_docker(self):
        cases = [
            (["main", "--streams", "0"], {}),
            (["main", "--tracking-capacity", "0"], {}),
            (["main", "--port", "65536"], {}),
            (["main", "--bind", "hostname"], {}),
            (["main"], {"TRACKING_EXECUTION_MODE": "local"}),
            (["main"], {"INFERENCE_EXECUTION_MODE": "local"}),
            (["main"], {"TRACKING_EXECUTION_MODE": "server"}),
            (["main"], {"INFERENCE_EXECUTION_MODE": "server"}),
            (["sam", "--gpu-count", "1"], {}),
            (["vlm", "--gpu-count", "1"], {}),
            (
                [
                    "sam",
                    "--gpu-id",
                    "0",
                    "--dry-run",
                    "--gpu-count",
                    "1",
                    "--streams",
                    "0",
                ],
                {},
            ),
            (["vlm", "--gpu-id", "2", "--dry-run", "--gpu-count", "1"], {}),
        ]
        for invalid in (
            "ftp://sam:8092",
            "http://user:secret@sam",
            "http://sam/v1/video/sessions",
            "http://sam?token=secret",
            "http://sam#x",
            "http://sam:0",
            "http://sam:70000",
        ):
            cases.append((["main", "--tracking-server", invalid], {}))
        for argv, settings in cases:
            with (
                self.subTest(argv=argv, settings=settings),
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    launcher.local, "read_local_settings", return_value=settings
                ),
                patch.object(launcher.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = launcher.main(argv)
            self.assertEqual(code, 2)
            run.assert_not_called()

    def test_sam_missing_empty_and_lfs_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "sam.pt"
            for content in (None, b"", b"version https://git-lfs.github.com/spec/v1"):
                if content is not None:
                    checkpoint.write_bytes(content)
                with (
                    self.subTest(content=content),
                    self.assertRaises(launcher.LaunchError),
                ):
                    launcher.validate_sam_weights(
                        {"SAM31_CHECKPOINT_FILE": str(checkpoint)}
                    )

    def test_saved_env_server_configuration_and_explicit_capacity_are_preserved(self):
        args = launcher.parser().parse_args(
            ["main", "--streams", "24", "--port", "9000"]
        )
        plan = launcher.main_environment(
            {
                "TRACKING_EXECUTION_MODE": "server",
                "TRACKING_SERVER_URL": "http://sam:8092/",
                "TRACKING_SERVER_CAPACITY": "16",
                "INFERENCE_SERVER_URL": "https://vlm.example/",
                "INFERENCE_SERVER_MODEL": "customer-qwen",
                "HARNESS_BIND_ADDRESS": "::1",
            },
            args,
        )
        self.assertEqual(plan["TRACKING_SERVER_CAPACITY"], "16")
        self.assertEqual(plan["HARNESS_FLEET_CAPACITY"], "24")
        self.assertEqual(plan["INFERENCE_EXECUTION_MODE"], "server")
        self.assertEqual(plan["INFERENCE_SERVER_MODEL"], "customer-qwen")
        self.assertEqual(plan["INFERENCE_SERVER_URL"], "https://vlm.example")
        self.assertEqual(plan["HARNESS_BIND_ADDRESS"], "[::1]")
        self.assertEqual(plan["HARNESS_PORT"], "9000")

    def test_launcher_settings_do_not_read_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                'TRACKING_SERVER_URL="http://sam:8092"\n'
                "TRACKING_SERVER_CAPACITY=16\n"
                "INFERENCE_API_KEY=cloud-secret\nLOCAL_INFERENCE_API_KEY=vlm-secret\n",
                encoding="utf-8",
            )
            self.assertEqual(
                launcher.local.read_local_settings(path, keys=launcher.SETTINGS),
                {
                    "TRACKING_SERVER_URL": "http://sam:8092",
                    "TRACKING_SERVER_CAPACITY": "16",
                },
            )

    @unittest.skipUnless(shutil.which("docker"), "Docker Compose not installed")
    def test_rendered_main_fleet_has_cpu_only_isolated_lanes_and_preserved_cloud_config(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            env_file = directory / ".env"
            env_file.write_text(
                "INFERENCE_API_KEY=cloud-test-key\nINFERENCE_PROVIDER=openrouter\n"
            )
            fleet_file = directory / "fleet.yaml"
            for count in (1, 2, 12, 13, 16, 100):
                args = launcher.parser().parse_args(
                    [
                        "main",
                        "--streams",
                        str(count),
                        "--tracking-server",
                        "http://sam:8092",
                        "--port",
                        "8888",
                    ]
                )
                fleet_file.write_text(
                    launcher.application_fleet(count), encoding="utf-8"
                )
                environment = dict(os.environ)
                environment.update(launcher.main_environment({}, args))
                environment["DOCKER_CONFIG"] = isolated_docker_config(Path(temporary))
                environment["HARNESS_STREAM_URL"] = "rtsp://example/first"
                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "--env-file",
                        str(env_file),
                        "-f",
                        str(ROOT / launcher.COMPOSE_FILES["main"]),
                        "-f",
                        str(fleet_file),
                        "config",
                        "--format",
                        "json",
                    ],
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                services = json.loads(result.stdout)["services"]
                self.assertEqual(len(services), count)
                for index in range(1, count + 1):
                    name = "harness" if index == 1 else f"harness-{index:02d}"
                    service = services[name]
                    self.assertNotIn("deploy", service)
                    self.assertNotIn("gpus", service)
                    self.assertNotIn("depends_on", service)
                    self.assertEqual(
                        service["build"].get("dockerfile", "Dockerfile"), "Dockerfile"
                    )
                    self.assertEqual(service["image"], "goodvision-harness:demo")
                    env = service["environment"]
                    self.assertEqual(env["FACE_PROVIDER"], "CPUExecutionProvider")
                    self.assertEqual(env["INFERENCE_PROVIDER"], "openrouter")
                    self.assertEqual(env["INFERENCE_API_KEY"], "cloud-test-key")
                    self.assertEqual(env["LOCAL_GPU_AVAILABLE"], "false")
                    self.assertEqual(env["TRACKING_SERVER_URL"], "http://sam:8092")
                    self.assertEqual(env["TRACKING_SERVER_CAPACITY"], str(count))
                    self.assertEqual(env["HARNESS_FLEET_CAPACITY"], str(count))
                    self.assertEqual(env["HARNESS_FLEET_CAMERA_INDEX"], str(index))
                    self.assertEqual(env["HARNESS_AUTOSTART"], "false")
                    volumes = {
                        item["target"]: item["source"] for item in service["volumes"]
                    }
                    self.assertEqual(volumes["/data"], f"camera-{index:02d}-data")
                    self.assertEqual(volumes["/models"], "face-models")
                    self.assertNotIn("/models/vision", volumes)
                    self.assertNotIn("/models/sam3.1_multiplex.pt", volumes)
                    if index == 1:
                        self.assertEqual(len(service["ports"]), 1)
                        self.assertEqual(service["ports"][0]["published"], "8888")
                        self.assertEqual(
                            env["HARNESS_STREAM_URL"], "rtsp://example/first"
                        )
                    else:
                        self.assertFalse(service.get("ports"))
                        self.assertEqual(env["HARNESS_STREAM_URL"], "")

    @unittest.skipUnless(shutil.which("docker"), "Docker Compose not installed")
    def test_rendered_workers_publish_only_their_service_and_requested_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text("LOCAL_INFERENCE_API_KEY=worker-test-key\n")
            for module, count in (("sam", 1), ("sam", 12), ("sam", 16), ("vlm", 1)):
                argv = [module, "--gpu-id", "3", "--bind", "192.0.2.12"]
                if module == "sam":
                    argv.extend(["--streams", str(count)])
                args = launcher.parser().parse_args(argv)
                environment = dict(os.environ)
                environment.update(
                    launcher.worker_environment(module, {}, args, DEVICES)
                )
                environment["DOCKER_CONFIG"] = isolated_docker_config(Path(temporary))
                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "--env-file",
                        str(env_file),
                        "-f",
                        str(ROOT / launcher.COMPOSE_FILES[module]),
                        "config",
                        "--format",
                        "json",
                    ],
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                services = json.loads(result.stdout)["services"]
                self.assertEqual(
                    set(services), {"vlm-server", "vlm-processing"}
                    if module == "vlm" else {"sam-server"},
                )
                service = services[f"{module}-server"]
                public_service = (
                    services["vlm-processing"] if module == "vlm" else service
                )
                self.assertEqual(public_service["ports"][0]["host_ip"], "192.0.2.12")
                gpu = service["deploy"]["resources"]["reservations"]["devices"]
                self.assertEqual(gpu[0]["device_ids"], ["3"])
                targets = {item["target"] for item in service["volumes"]}
                if module == "sam":
                    self.assertEqual(
                        service["environment"]["SAM31_LIVE_CAMERAS"], str(count)
                    )
                    self.assertEqual(targets, {"/models/sam3.1_multiplex.pt"})
                    self.assertIn(
                        "SAM31_LIVE_CAMERAS", service["healthcheck"]["test"][-1]
                    )
                else:
                    processing = services["vlm-processing"]
                    self.assertFalse(service.get("ports"))
                    self.assertNotIn("deploy", processing)
                    self.assertEqual(processing["ports"][0]["target"], 8080)
                    self.assertEqual(
                        processing["environment"]["LOCAL_INFERENCE_API_KEY"],
                        "worker-test-key",
                    )
                    self.assertEqual(
                        processing["depends_on"]["vlm-server"]["condition"],
                        "service_healthy",
                    )
                    self.assertEqual(targets, {"/models/vision"})
                    self.assertEqual(
                        service["environment"]["VLLM_API_KEY"], "worker-test-key"
                    )
                    self.assertEqual(service["image"], "vllm/vllm-openai:v0.17.1")
                    self.assertNotIn("--reasoning-parser", service["command"])


if __name__ == "__main__":
    unittest.main()
