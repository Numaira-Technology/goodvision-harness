from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "face_gpu_runtime_check.py"
ORT_EXCEPTION = "insightface 1.0.1 requires onnxruntime, which is not installed."
CV_EXCEPTION = "insightface 1.0.1 requires opencv-python, which is not installed."
VALID_OUTPUT = f"{ORT_EXCEPTION}\n{CV_EXCEPTION}\n"


def _load_runtime_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tested_face_gpu_runtime_check", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime_check() -> ModuleType:
    return _load_runtime_check()


def test_runtime_check_import_does_not_import_inference_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    inference_modules = {"cv2", "insightface", "numpy", "onnxruntime"}

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] in inference_modules:
            raise AssertionError(
                f"unexpected import while loading audit helper: {name}"
            )
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", guarded_import)
        module = _load_runtime_check()
    assert callable(module.validate_dependency_audit)


@pytest.mark.parametrize(
    "output",
    [
        VALID_OUTPUT,
        f"{CV_EXCEPTION}\n{ORT_EXCEPTION}",
        f"\n  {ORT_EXCEPTION}\r\n\t{CV_EXCEPTION}  \r\n\n",
    ],
    ids=["declared-order", "reverse-order", "blank-lines-and-whitespace"],
)
def test_dependency_audit_accepts_only_the_two_declared_exceptions(
    runtime_check: ModuleType, output: str,
) -> None:
    assert runtime_check.validate_dependency_audit(1, output) is None


@pytest.mark.parametrize(
    "output",
    [
        "",
        "\n  \n",
        ORT_EXCEPTION,
        CV_EXCEPTION,
        f"{ORT_EXCEPTION}\n{ORT_EXCEPTION}",
        f"{CV_EXCEPTION}\n{CV_EXCEPTION}",
        VALID_OUTPUT + ORT_EXCEPTION,
        VALID_OUTPUT + CV_EXCEPTION,
        VALID_OUTPUT
        + "some-package 1.0 requires missing-package, which is not installed.",
        "unknown dependency error",
        "No broken requirements found.",
        f"{ORT_EXCEPTION}\n"
        "insightface 1.0.0 requires opencv-python, which is not installed.",
        f"{ORT_EXCEPTION}\n"
        "insightface 1.0.1 requires opencv-python>=6, which is not installed.",
    ],
    ids=[
        "empty", "whitespace-only", "missing-opencv", "missing-ort",
        "duplicate-ort-only", "duplicate-opencv-only", "extra-duplicate-ort",
        "extra-duplicate-opencv", "extra-unrelated-error", "unknown-error",
        "unexpected-clean-check", "changed-insightface-version",
        "changed-dependency-requirement",
    ],
)
def test_dependency_audit_rejects_missing_extra_duplicate_or_unknown_lines(
    runtime_check: ModuleType, output: str,
) -> None:
    with pytest.raises(RuntimeError, match="Unexpected dependency audit result"):
        runtime_check.validate_dependency_audit(1, output)


@pytest.mark.parametrize("returncode", [0, -1, 2, 127, 137])
def test_dependency_audit_rejects_wrong_returncode_even_with_expected_lines(
    runtime_check: ModuleType, returncode: int,
) -> None:
    with pytest.raises(RuntimeError, match="Unexpected dependency audit result"):
        runtime_check.validate_dependency_audit(returncode, VALID_OUTPUT)
