"""Launch-plan regression checks without a GPU, model downloads, or Docker writes."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

from compose_support import isolated_docker_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "local_gpu_launcher", ROOT / "tools" / "start_local_gpu.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def devices(count):
    return [
        {"index": index, "name": "NVIDIA L40S", "memory_mib": 46068}
        for index in range(count)
    ]


def write_offline_model(directory):
    for name, value in {
        "config.json": {
            "model_type": "qwen3_vl_moe",
            "quantization_config": {"quant_method": "fp8"},
        },
        "preprocessor_config.json": {"image_processor_type": "Qwen3VLImageProcessor"},
        "video_preprocessor_config.json": {"video_processor_type": "Qwen3VLProcessor"},
        "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
        "tokenizer.json": {"version": "1.0"},
        "chat_template.json": {"chat_template": "{{ messages }}"},
    }.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    (directory / "model-1.safetensors").write_bytes(b"mock weights")
    (directory / "sam.pt").write_bytes(b"mock checkpoint")


class LocalGpuLauncherTests(unittest.TestCase):
    def test_single_device_shares_and_leaves_memory_for_tracking(self):
        plan = launcher.plan_environment(devices(1), {})
        self.assertEqual(plan["LOCAL_GPU_COUNT"], "1")
        self.assertEqual(plan["LOCAL_INFERENCE_GPU_ID"], "0")
        self.assertEqual(plan["LOCAL_TRACKING_GPU_ID"], "0")
        self.assertEqual(plan["LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION"], "0.80")
        self.assertEqual(plan["LOCAL_INFERENCE_MAX_NUM_SEQS"], "4")
        self.assertEqual(plan["LOCAL_INFERENCE_EXECUTION_MODE"], "--enforce-eager")
        self.assertTrue(plan["LOCAL_INFERENCE_MODEL_DIRECTORY"].endswith(
            "qwen3-vl-30b-a3b-instruct-fp8",
        ))

    def test_two_and_more_devices_always_isolate_model_and_other_work(self):
        for count in (2, 4):
            with self.subTest(count=count):
                plan = launcher.plan_environment(devices(count), {})
                self.assertEqual(plan["LOCAL_GPU_COUNT"], str(count))
                self.assertEqual(plan["LOCAL_INFERENCE_GPU_ID"], "0")
                self.assertEqual(plan["LOCAL_TRACKING_GPU_ID"], "1")
                self.assertEqual(plan["LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION"], "0.90")
                self.assertTrue(
                    plan["LOCAL_INFERENCE_MODEL_DIRECTORY"].endswith(
                        "qwen3-vl-30b-a3b-instruct-fp8",
                    )
                )
                self.assertEqual(plan["LOCAL_INFERENCE_MAX_NUM_SEQS"], "8")
                self.assertEqual(plan["LOCAL_INFERENCE_MAX_BATCHED_TOKENS"], "4096")

    def test_inventory_uses_actual_indices_and_rejects_invalid_output(self):
        inventory = launcher.parse_nvidia_devices(
            "3, NVIDIA L40S, 46068\n1, NVIDIA L40S, 46068\n"
        )
        plan = launcher.plan_environment(inventory, {})
        self.assertEqual(plan["LOCAL_INFERENCE_GPU_ID"], "1")
        self.assertEqual(plan["LOCAL_TRACKING_GPU_ID"], "3")
        for output in (
            "",
            "AMD Radeon",
            "0, NVIDIA, unknown",
            "0, NVIDIA, 1\n0, NVIDIA, 1",
        ):
            with self.subTest(output=output), self.assertRaises(launcher.LaunchError):
                launcher.parse_nvidia_devices(output)

    def test_no_nvidia_does_not_fall_back_to_another_vendor(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 15)):
            with patch.object(launcher.subprocess, "run", side_effect=error):
                with self.assertRaises(launcher.LaunchError):
                    launcher.detect_nvidia_devices()

    def test_model_and_tracker_switches_are_independent_and_cloud_keys_excluded(self):
        for inference in ("true", "false"):
            for tracking in ("true", "false"):
                plan = launcher.plan_environment(
                    devices(2),
                    {
                        "LOCAL_INFERENCE_ENABLED": inference,
                        "LOCAL_TRACKING_ENABLED": tracking,
                        "INFERENCE_API_KEY": "do-not-print",
                        "LOCAL_INFERENCE_MODEL": "customer-vision",
                    },
                )
                self.assertEqual(plan["LOCAL_INFERENCE_ENABLED"], inference)
                self.assertEqual(plan["LOCAL_TRACKING_ENABLED"], tracking)
                self.assertEqual(plan["LOCAL_INFERENCE_MODEL"], "customer-vision")
                self.assertNotIn("INFERENCE_API_KEY", plan)

    def test_invalid_memory_and_model_limits_fail_before_docker(self):
        for settings in (
            {"LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION": "nan"},
            {"LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION": "1"},
            {"LOCAL_TRACKING_ENABLED": "maybe"},
            {"LOCAL_INFERENCE_MAX_MODEL_LEN": "ten"},
            {"LOCAL_INFERENCE_MAX_TOKENS": "9000"},
            {
                "LOCAL_INFERENCE_MAX_MODEL_LEN": "32768",
                "LOCAL_INFERENCE_MAX_TOKENS": "20000",
            },
            {"LOCAL_INFERENCE_MAX_NUM_SEQS": "0"},
            {"LOCAL_INFERENCE_MAX_BATCHED_TOKENS": "0"},
            {"LOCAL_INFERENCE_EXECUTION_MODE": "--arbitrary-server-flag"},
            {"LOCAL_TRACKING_FPS": "nan"},
        ):
            with (
                self.subTest(settings=settings),
                self.assertRaises(launcher.LaunchError),
            ):
                launcher.plan_environment(devices(2), settings)

    def test_env_reader_ignores_secrets_and_accepts_quoted_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                "INFERENCE_API_KEY=do-not-print\n"
                'LOCAL_INFERENCE_MODEL_DIRECTORY="D:/Model Files/vision" # local\n'
                "export LOCAL_TRACKING_ENABLED=false\n",
                encoding="utf-8",
            )
            settings = launcher.read_local_settings(path)
            self.assertEqual(
                settings,
                {
                    "LOCAL_INFERENCE_MODEL_DIRECTORY": "D:/Model Files/vision",
                    "LOCAL_TRACKING_ENABLED": "false",
                },
            )
            path.write_text("LOCAL_INFERENCE_MODEL_DIRECTORY=${MODELS}/vision\n")
            with self.assertRaises(launcher.LaunchError):
                launcher.read_local_settings(path)

    def test_command_selects_only_local_overlay_and_explicit_services(self):
        command = launcher.compose_command(
            root=ROOT, env_file=ROOT / ".env", with_paddle=True, build=False
        )
        self.assertIn(str(ROOT / "compose.local-gpu.yaml"), command)
        self.assertNotIn("--build", command)
        self.assertEqual(
            command[-5:],
            [
                "harness",
                "local-inference",
                "local-vlm-processing",
                "local-tracking",
                "rich-label-worker",
            ],
        )
        self.assertIn("--wait", command)
        self.assertIn("rich", command)

    def test_dry_run_has_no_process_or_file_mutations(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(launcher, "read_local_settings", return_value={}):
                with patch.object(launcher.subprocess, "run") as run:
                    with patch.object(launcher, "validate_weights") as validate:
                        with contextlib.redirect_stdout(output):
                            status = launcher.main(["--dry-run", "--gpu-count", "2"])
        self.assertEqual(status, 0)
        run.assert_not_called()
        validate.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["environment"]["LOCAL_GPU_COUNT"], "2")
        self.assertEqual(plan["streams"], 1)
        self.assertEqual(plan["provisioned_lanes"], 1)
        self.assertEqual(plan["console_url"], "http://127.0.0.1:8080")
        self.assertEqual(plan["environment"]["HARNESS_FLEET_CAPACITY"], "1")
        self.assertIsNone(plan["fleet_overlay"])
        self.assertNotIn("harness-02", plan["command"])

    def test_every_supported_camera_count_selects_only_requested_services(self):
        for streams in (*range(1, 13), 13, 16, 24, 64, 100):
            output = io.StringIO()
            with (
                self.subTest(streams=streams),
                patch.dict(os.environ, {}, clear=True),
                patch.object(launcher, "read_local_settings", return_value={}),
                patch.object(launcher.subprocess, "run") as run,
                contextlib.redirect_stdout(output),
            ):
                status = launcher.main(
                    [
                        "--dry-run",
                        "--gpu-count",
                        "2",
                        "--streams",
                        str(streams),
                    ]
                )
            self.assertEqual(status, 0)
            run.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["streams"], streams)
            self.assertEqual(plan["provisioned_lanes"], streams)
            self.assertEqual(plan["console_url"], "http://127.0.0.1:8080")
            self.assertEqual(
                plan["environment"]["HARNESS_FLEET_CAPACITY"], str(streams)
            )
            services = [
                item
                for item in plan["command"]
                if item == "harness" or item.startswith("harness-")
            ]
            self.assertEqual(
                services,
                ["harness"]
                + [f"harness-{index:02d}" for index in range(2, streams + 1)],
            )
            self.assertEqual(plan["fleet_overlay"] is None, streams == 1)

    def test_camera_count_outside_supported_range_is_rejected(self):
        for streams in (0, -1):
            with (
                self.subTest(streams=streams),
                patch.object(launcher.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = launcher.main(
                    [
                        "--dry-run",
                        "--gpu-count",
                        "2",
                        "--streams",
                        str(streams),
                    ]
                )
            self.assertEqual(status, 2)
            run.assert_not_called()

    def test_fleet_limits_and_service_selection(self):
        for streams in (0, -1, 1):
            with self.assertRaises(launcher.LaunchError):
                launcher.fleet_overlay(streams)
        for port in (0, 65536, True):
            with self.assertRaises(launcher.LaunchError):
                launcher.fleet_overlay(12, port=port)
        # N no longer consumes N public ports; even the last TCP port is valid.
        overlay = launcher.fleet_overlay(100, port=65535)
        self.assertIn('"published": "65535"', overlay)
        self.assertEqual(overlay.count('"published"'), 1)
        self.assertEqual(overlay.count("ports: !override []"), 99)
        self.assertIn("harness-100:", overlay)
        command = launcher.compose_command(
            root=ROOT,
            env_file=None,
            with_paddle=False,
            build=False,
            streams=12,
            fleet_file=ROOT / ".goodvision/local-gpu-compose.yaml",
        )
        self.assertIn("harness-12", command)
        self.assertNotIn("rich-label-worker", command)

    def test_console_port_override_applies_to_single_and_fleet_plans(self):
        for streams in (1, 24):
            for explicit, expected in (
                ([], "9000"),
                (["--base-port", "65535"], "65535"),
            ):
                output = io.StringIO()
                with (
                    self.subTest(streams=streams, explicit=explicit),
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(
                        launcher,
                        "read_local_settings",
                        return_value={
                            "HARNESS_PORT": "9000",
                            "HARNESS_BIND_ADDRESS": "0.0.0.0",
                        },
                    ),
                    patch.object(launcher.subprocess, "run") as run,
                    contextlib.redirect_stdout(output),
                ):
                    status = launcher.main(
                        [
                            "--dry-run",
                            "--gpu-count",
                            "2",
                            "--streams",
                            str(streams),
                            *explicit,
                        ]
                    )
                self.assertEqual(status, 0)
                run.assert_not_called()
                plan = json.loads(output.getvalue())
                self.assertEqual(plan["environment"]["HARNESS_PORT"], expected)
                self.assertEqual(plan["console_url"], f"http://127.0.0.1:{expected}")

    def test_invalid_console_port_fails_before_docker(self):
        for value in ("0", "65536", "not-a-port"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    launcher,
                    "read_local_settings",
                    return_value={
                        "HARNESS_PORT": value,
                    },
                ),
                patch.object(launcher.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = launcher.main(["--dry-run", "--gpu-count", "2"])
            self.assertEqual(status, 2)
            run.assert_not_called()

    @unittest.skipUnless(shutil.which("docker"), "Docker Compose not installed")
    def test_rendered_fleet_keeps_ports_data_models_and_device_isolation(self):
        # docker compose config is read-only: no daemon, builds or model downloads.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            env_file = directory / ".env"
            env_file.write_text("")
            fleet_file = directory / "fleet.yaml"
            for streams, count in product((*range(2, 13), 13, 16, 24), (1, 2)):
                fleet_file.write_text(launcher.fleet_overlay(streams), encoding="utf-8")
                environment = dict(os.environ)
                environment.update(launcher.plan_environment(devices(count), {}))
                environment["DOCKER_CONFIG"] = isolated_docker_config(directory)
                command = [
                    "docker",
                    "compose",
                    "--env-file",
                    str(env_file),
                    "-f",
                    str(ROOT / "compose.yaml"),
                    "-f",
                    str(ROOT / "compose.local-gpu.yaml"),
                    "-f",
                    str(fleet_file),
                    "--profile",
                    "rich",
                    "config",
                    "--format",
                    "json",
                ]
                result = subprocess.run(
                    command,
                    env=environment,
                    text=True,
                    capture_output=True,
                    # Windows CI can spend over 30 seconds running Compose.
                    # This checks rendered configuration, not CLI latency.
                    timeout=60,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                services = json.loads(result.stdout)["services"]
                self.assertEqual(len(services), streams + 4)
                processing = services["local-vlm-processing"]
                self.assertFalse(processing.get("ports"))
                self.assertNotIn("deploy", processing)
                self.assertEqual(
                    processing["depends_on"]["local-inference"]["condition"],
                    "service_healthy",
                )
                tracking_device = "0" if count == 1 else "1"
                worker = services["local-tracking"]
                self.assertEqual(worker["image"], "goodvision-contour-runtime:0.1.0")
                self.assertEqual(
                    worker["build"]["dockerfile"], "contour/Dockerfile.engine",
                )
                self.assertEqual(
                    worker["environment"]["SAM31_FIXED_PERSON_TEXT"], "true",
                )
                self.assertTrue(worker["build"]["context"].replace("\\", "/").endswith(
                    "/workers",
                ))
                self.assertEqual(len(worker["volumes"]), 1)
                self.assertEqual(worker["volumes"][0]["target"], "/models/contour.pt")
                self.assertTrue(worker["volumes"][0]["read_only"])
                for index in range(1, streams + 1):
                    name = "harness" if index == 1 else f"harness-{index:02d}"
                    service = services[name]
                    if index == 1:
                        self.assertEqual(len(service["ports"]), 1)
                        self.assertEqual(service["ports"][0]["published"], "8080")
                    else:
                        self.assertEqual(service.get("ports", []), [])
                    volumes = {item["target"]: item for item in service["volumes"]}
                    self.assertEqual(
                        volumes["/data"]["source"], f"camera-{index:02d}-data"
                    )
                    self.assertEqual(volumes["/models"]["source"], "face-models")
                    env = service["environment"]
                    self.assertEqual(env["LOCAL_INFERENCE_INSPECTION_ENDPOINT"], "http://local-vlm-processing:8080/v1/inspections")
                    self.assertIn("LOCAL_INFERENCE_API_KEY", env)
                    self.assertIn("INFERENCE_SERVER_PROTOCOL", env)
                    self.assertEqual(env["HARNESS_FLEET_CAPACITY"], str(streams))
                    self.assertEqual(env["HARNESS_FLEET_CAMERA_INDEX"], str(index))
                    self.assertEqual(
                        env["HARNESS_FLEET_PRIMARY_URL"], "http://harness:8080"
                    )
                    self.assertEqual(env["HARNESS_SOURCE_ID"], f"camera-{index:02d}")
                    self.assertEqual(env["HARNESS_AUTOSTART"], "false")
                    self.assertIn(
                        f"/cameras/camera-{index:02d}/", env["LOCAL_TRACKING_ENDPOINT"]
                    )
                for name, service in services.items():
                    if name == "local-vlm-processing":
                        continue
                    self.assertIn(
                        "devices", service["deploy"]["resources"]["reservations"], name,
                    )
                    reservation = service["deploy"]["resources"]["reservations"][
                        "devices"
                    ]
                    expected = "0" if name == "local-inference" else tracking_device
                    self.assertEqual(len(reservation), 1)
                    self.assertEqual(reservation[0]["device_ids"], [expected])
                    self.assertNotIn("count", reservation[0])
                self.assertEqual(
                    services["local-tracking"]["environment"]["SAM31_LIVE_CAMERAS"],
                    str(streams),
                )
                self.assertIn(
                    "/health/cameras/camera-01/perception/sam_live",
                    services["local-tracking"]["healthcheck"]["test"][-1],
                )
                command = services["local-inference"]["command"]
                self.assertEqual(
                    command[command.index("--tensor-parallel-size") + 1], "1"
                )
                self.assertEqual(
                    command[command.index("--max-num-seqs") + 1],
                    "4" if count == 1 else "8",
                )
                self.assertEqual(
                    command[command.index("--max-num-batched-tokens") + 1], "4096",
                )
                self.assertNotIn("--reasoning-parser", command)
                self.assertNotIn("--default-chat-template-kwargs", command)

    def test_simulated_devices_cannot_start_containers(self):
        with patch.object(launcher.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()):
                status = launcher.main(["--gpu-count", "2"])
        self.assertEqual(status, 2)
        run.assert_not_called()

    def test_missing_or_incomplete_offline_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = launcher.plan_environment(
                devices(2),
                {
                    "LOCAL_INFERENCE_MODEL_DIRECTORY": str(directory),
                    "SAM31_CHECKPOINT_FILE": str(directory / "sam.pt"),
                },
            )
            with self.assertRaises(launcher.LaunchError):
                launcher.validate_weights(plan)
            write_offline_model(directory)
            launcher.validate_weights(plan)
            (directory / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-1.safetensors",
                            "b": "missing.safetensors",
                        },
                    }
                )
            )
            with self.assertRaises(launcher.LaunchError):
                launcher.validate_weights(plan)

    def test_offline_bundle_requires_processor_tokenizer_and_chat_template(self):
        for name in (
            "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
            "chat_template.json", "video_preprocessor_config.json",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                write_offline_model(directory)
                (directory / name).unlink()
                plan = launcher.plan_environment(devices(2), {
                    "LOCAL_INFERENCE_MODEL_DIRECTORY": str(directory),
                    "SAM31_CHECKPOINT_FILE": str(directory / "sam.pt"),
                })
                with self.assertRaises(launcher.LaunchError):
                    launcher.validate_weights(plan)

    def test_invalid_tokenizer_json_or_chat_template_is_rejected(self):
        for name, payload in (
            ("tokenizer.json", "not JSON"),
            ("chat_template.json", '{"chat_template":123}'),
            ("chat_template.json", '{"chat_template":"  "}'),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                write_offline_model(directory)
                (directory / name).write_text(payload)
                plan = launcher.plan_environment(devices(2), {
                    "LOCAL_INFERENCE_MODEL_DIRECTORY": str(directory),
                    "SAM31_CHECKPOINT_FILE": str(directory / "sam.pt"),
                })
                with self.assertRaises(launcher.LaunchError):
                    launcher.validate_weights(plan)

    def test_sharded_download_without_index_or_with_lfs_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_offline_model(directory)
            weight = directory / "model-1.safetensors"
            weight.rename(directory / "model-00001-of-00004.safetensors")
            plan = launcher.plan_environment(devices(2), {
                "LOCAL_INFERENCE_MODEL_DIRECTORY": str(directory),
                "SAM31_CHECKPOINT_FILE": str(directory / "sam.pt"),
            })
            with self.assertRaisesRegex(launcher.LaunchError, "index"):
                launcher.validate_weights(plan)
            (directory / "model-00001-of-00004.safetensors").rename(weight)
            weight.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(launcher.LaunchError, "LFS pointer"):
                launcher.validate_weights(plan)

    def test_weights_exceeding_the_selected_card_budget_fail_before_docker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_offline_model(directory)
            with (directory / "model-1.safetensors").open("wb") as weight:
                weight.write(b"fixture")
                weight.truncate(2 * 1024**2)
            inventory = [{"index": 3, "memory_mib": 4, "name": "test GPU"}]
            plan = launcher.plan_environment(inventory, {
                "LOCAL_INFERENCE_MODEL_DIRECTORY": str(directory),
                "SAM31_CHECKPOINT_FILE": str(directory / "sam.pt"),
            })
            launcher.validate_weights(plan, inventory)
            plan["LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION"] = "0.45"
            with self.assertRaisesRegex(launcher.LaunchError, "memory budget"):
                launcher.validate_weights(plan, inventory)


if __name__ == "__main__":
    unittest.main()
