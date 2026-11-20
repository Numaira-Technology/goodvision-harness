"""Offline launch parity using only a locally created stdlib venv and test code."""

import copy
import json
import os
import py_compile
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from tools import env_source as es
from tools import registry

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux packed env layout")


@pytest.fixture
def launch_env(tmp_path):
    prefix = tmp_path / "packed env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)], check=True
    )
    env = copy.deepcopy(registry.env_of("videojev"))
    env["unpack_to"] = str(prefix)
    return env


def _run_pair(env, args, cwd):
    argv, child = es.launch_argv(env, *args)
    direct = subprocess.run(
        [es.packed_python(env), "-s", *args],
        cwd=cwd,
        env=child,
        capture_output=True,
        text=True,
        timeout=30,
    )
    wrapped = subprocess.run(
        argv,
        cwd=cwd,
        env=child,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert direct.returncode == 0, direct.stderr
    assert wrapped.returncode == 0, wrapped.stderr
    return direct, wrapped


_CONTEXT_PROBE = textwrap.dedent("""\
    import atexit, json, pickle, site, sys

    def worker():
        return 42

    def report(phase):
        print(json.dumps({
            'phase': phase,
            'argv': sys.argv,
            'name': __name__,
            'file': globals().get('__file__'),
            'package': __package__,
            'spec': __spec__.name if __spec__ else None,
            'path': sys.path,
            'main_namespace': vars(sys.modules['__main__']) is globals(),
            'pickled': pickle.loads(pickle.dumps(worker)) is worker,
            'prefix': sys.prefix,
            'executable': sys.executable,
            'user_site': site.ENABLE_USER_SITE,
            'sibling': value,
        }))

    if __name__ == '__main__':
        report('running')
        atexit.register(report, 'shutdown')
    """)


@pytest.mark.parametrize(
    "mode",
    [
        "module",
        "package_module",
        "code",
        "script",
        "relative_script",
        "symlink_script",
        "directory",
        "zip",
        "bytecode",
    ],
)
def test_launch_context_matches_direct_python(launch_env, tmp_path, mode):
    cwd = tmp_path / "working"
    cwd.mkdir()
    scripts = tmp_path / "scripts with spaces"
    scripts.mkdir()
    (cwd / "sibling.py").write_text("value = 99\n")
    (scripts / "sibling.py").write_text("value = 42\n")
    source = "from sibling import value\n" + _CONTEXT_PROBE
    script = scripts / "entry.py"
    script.write_text(source)
    if mode == "module":
        (cwd / "entry.py").write_text(source)
        args = ["-m", "entry"]
    elif mode == "package_module":
        package = cwd / "package"
        package.mkdir()
        (package / "__init__.py").touch()
        (package / "sibling.py").write_text("value = 17\n")
        (package / "entry.py").write_text(
            "from .sibling import value\n" + _CONTEXT_PROBE
        )
        args = ["-m", "package.entry"]
    elif mode == "code":
        args = ["-c", source]
    elif mode == "relative_script":
        args = [os.path.relpath(script, cwd)]
    elif mode == "symlink_script":
        alias = cwd / "alias.py"
        alias.symlink_to(script)
        args = [str(alias)]
    elif mode == "directory":
        (scripts / "__main__.py").write_text(source)
        args = [str(scripts)]
    elif mode == "zip":
        archive = tmp_path / "entry.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("__main__.py", source)
            bundle.writestr("sibling.py", "value = 23\n")
        args = [str(archive)]
    elif mode == "bytecode":
        bytecode = scripts / "entry.pyc"
        py_compile.compile(str(script), cfile=str(bytecode), doraise=True)
        args = [str(bytecode)]
    else:
        args = [str(script)]
    args += ["space value", "$(literal);'quoted", "--flag"]
    direct, wrapped = _run_pair(launch_env, args, cwd)
    expected = [json.loads(line) for line in direct.stdout.splitlines()]
    actual = [json.loads(line) for line in wrapped.stdout.splitlines()]
    assert [row["phase"] for row in expected] == ["running", "shutdown"]
    assert all(row["main_namespace"] and row["pickled"] for row in expected)
    assert not direct.stderr and not wrapped.stderr
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        differences = {
            key: (reference.get(key), observed.get(key))
            for key in reference.keys() | observed.keys()
            if reference.get(key) != observed.get(key)
        }
        assert not differences, (reference["phase"], differences)


@pytest.mark.parametrize("mode", ["module", "code", "script", "script_local"])
def test_launch_spawn_matches_direct_python(launch_env, tmp_path, mode):
    cwd = tmp_path / "working"
    cwd.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    worker = textwrap.dedent("""\
        def worker():
            import json, site, sys
            print(json.dumps({
                'prefix': sys.prefix, 'user_site': site.ENABLE_USER_SITE,
            }))
        """)
    main = textwrap.dedent("""\
        if __name__ == '__main__':
            import multiprocessing
            process = multiprocessing.get_context('spawn').Process(target=worker)
            process.start()
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join()
                raise SystemExit('spawn did not finish')
            assert process.exitcode == 0, process.exitcode
        """)
    if mode == "code":
        # Native -c cannot respawn its own local definitions; use an importable worker.
        (cwd / "support.py").write_text(worker)
        args = ["-c", "from support import worker\n" + main]
    elif mode == "module":
        (cwd / "entry.py").write_text(worker + main)
        args = ["-m", "entry"]
    else:
        (scripts / "support.py").write_text(worker)
        script = scripts / "entry.py"
        source = "from support import worker\n" if mode == "script" else worker
        script.write_text(source + main)
        args = [str(script)]
    direct, wrapped = _run_pair(launch_env, args, cwd)
    assert not direct.stderr and not wrapped.stderr
    assert json.loads(wrapped.stdout) == json.loads(direct.stdout)


@pytest.mark.parametrize("change", ["prefix", "user_site"])
def test_launch_checks_the_actual_process_before_payload(launch_env, tmp_path, change):
    marker = tmp_path / "payload-ran"
    payload = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    argv, child = es.launch_argv(launch_env, "-c", payload)
    if change == "prefix":
        child["GOODVISION_ENV_PREFIX"] = str(tmp_path / "different")
        message = "sys.prefix"
    else:
        site_packages = next(
            Path(launch_env["unpack_to"]).glob("lib/python*/site-packages")
        )
        # Change startup only after preflight has succeeded, using known local code.
        (site_packages / "sitecustomize.py").write_text(
            "import site\nsite.ENABLE_USER_SITE = True\n"
        )
        message = "user site must be disabled"
    proc = subprocess.run(argv, env=child, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert message in proc.stderr
    assert not marker.exists()


def test_cwd_cannot_shadow_launcher_dependencies(launch_env, tmp_path):
    (tmp_path / "pkgutil.py").write_text("raise RuntimeError('cwd pkgutil executed')\n")
    direct, wrapped = _run_pair(launch_env, ["-c", "print('payload')"], tmp_path)
    assert direct.stdout == wrapped.stdout == "payload\n"
    assert not direct.stderr and not wrapped.stderr
