"""Package marker. Arms the HF Qwen rental log without editing that launcher.

``python -m tools.start_qwen8b_r2`` is the one renter that must not call
``vast_run_log.attach`` itself. Every other Vast rental calls ``attach``
at the start of ``main``. The hook does not change offer selection or scoring.
"""

from __future__ import annotations

import sys


def _install_run_log_hook() -> None:
    import runpy

    current = runpy._run_code
    if getattr(current, "_gv_run_log", False):
        return
    original = current

    def _run_code(
        code,
        run_globals,
        init_globals=None,
        mod_name=None,
        mod_spec=None,
        pkg_name=None,
        script_name=None,
    ):
        filename = getattr(code, "co_filename", "") or ""
        if mod_name == "__main__":
            try:
                from tools.vast_run_log import arm_if_launcher

                arm_if_launcher(filename)
            except Exception as exc:
                try:
                    sys.stderr.write(
                        f"run log: arm failed ({type(exc).__name__}); eval continues\n"
                    )
                    sys.stderr.flush()
                except (OSError, ValueError):
                    pass
        return original(
            code,
            run_globals,
            init_globals,
            mod_name,
            mod_spec,
            pkg_name,
            script_name,
        )

    _run_code._gv_run_log = True  # type: ignore[attr-defined]
    runpy._run_code = _run_code


_install_run_log_hook()
