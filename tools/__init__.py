"""Arm hardware logs without editing frozen launcher and native runner sources.

``python -m tools.start_qwen8b_r2`` is the one renter that must not call
``vast_run_log.attach`` itself. Every other Vast rental calls ``attach``
at the start of ``main``. Native GPU CLI runs finish their log when the module
returns or raises, preserving the outcome. The hook does not change scoring.
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
        native_session = None
        if mod_name == "__main__":
            try:
                from tools.vast_run_log import arm_if_launcher

                native_session = arm_if_launcher(filename)
            except Exception as exc:
                try:
                    sys.stderr.write(
                        f"run log: arm failed ({type(exc).__name__}); eval continues\n"
                    )
                    sys.stderr.flush()
                except (OSError, ValueError):
                    pass
        exit_code = 0
        try:
            return original(
                code,
                run_globals,
                init_globals,
                mod_name,
                mod_spec,
                pkg_name,
                script_name,
            )
        except SystemExit as exc:
            exit_code = (
                exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
            )
            raise
        except BaseException as exc:
            exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            raise
        finally:
            if native_session is not None:
                try:
                    from tools.vast_run_log import finish_native

                    finish_native(native_session, exit_code)
                except Exception:
                    # Recording must never mask the original CLI exception.
                    pass

    _run_code._gv_run_log = True  # type: ignore[attr-defined]
    runpy._run_code = _run_code


_install_run_log_hook()
