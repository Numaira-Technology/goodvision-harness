"""Offline restore handoff identity: no SSH, R2, pack execution or downloads."""

import copy
import json

import pytest

from tools import cloud_env, env_source, registry


def expected_identity():
    env = registry.env_of("videojev")
    return {
        key: copy.deepcopy(env[key])
        for key in (*env_source.IDENTITY_FIELDS, "manifest_key")
    }


def reply(identity):
    env = registry.env_of("videojev")
    return {
        "name": "videojev",
        "source": "r2",
        "activate": env["unpack_to"] + "/" + env["activate_rel"],
        "python": env["unpack_to"] + "/bin/python",
        "identity": identity,
    }


def restore(payload, cleanup):
    return cloud_env.restore_env(
        ssh_options=[], host="synthetic.invalid", port=22, name="videojev",
        r2_env={key: "synthetic" for key in cloud_env.R2_KEYS},
        uploader=lambda *args: None,
        runner=lambda *args, **kwargs: payload,
        cleaner=lambda *args: cleanup.append(args),
    )


@pytest.mark.parametrize("source", ["local", "r2"])
def test_reader_echoes_selected_pack_identity(monkeypatch, source):
    env = registry.env_of("videojev")
    monkeypatch.setattr(
        env_source, "_restore",
        lambda *args, **kwargs: (env["unpack_to"] + "/bin/activate", source),
    )
    monkeypatch.setattr(
        env_source, "packed_python", lambda unused: env["unpack_to"] + "/bin/python",
    )
    info = env_source.ensure_env("videojev")
    assert info["identity"] == expected_identity()
    cleanup = []
    assert restore(json.dumps(info), cleanup) == info["python"]
    assert len(cleanup) == 1


@pytest.mark.parametrize("change", [
    {"sha256": "a" * 64},
    {"size": 123},
    {"r2_key": "envs/videojev/another.tar.gz",
     "manifest_key": "envs/videojev/another.json"},
    {"manifest_key": "envs/videojev/wrong.json"},
    {"compat": {"python": "3.12", "cuda": "12.8",
                "torch": "9.9.9", "gpu_family": "ada"}},
    {"unpack_to": "/opt/env/other"},
    {"activate_rel": "bin/other"},
])
def test_same_path_reply_cannot_acknowledge_another_pack(change):
    identity = expected_identity() | change
    cleanup = []
    with pytest.raises(RuntimeError, match="identity"):
        restore(json.dumps(reply(identity)), cleanup)
    assert len(cleanup) == 1


@pytest.mark.parametrize("malformed", [None, [], "missing", "float-size"])
def test_missing_or_malformed_identity_is_not_legacy_success(malformed):
    identity = expected_identity()
    payload = reply(identity)
    if malformed == "missing":
        payload.pop("identity")
    elif malformed == "float-size":
        identity["size"] = float(identity["size"])
    else:
        payload["identity"] = malformed
    cleanup = []
    with pytest.raises(RuntimeError, match="identity"):
        restore(json.dumps(payload), cleanup)
    assert len(cleanup) == 1


def test_last_wrong_identity_cannot_fall_back_to_an_earlier_success():
    identity = expected_identity()
    payload = json.dumps(reply(identity)) + "\n" + json.dumps(
        reply(identity | {"sha256": "a" * 64})
    )
    cleanup = []
    with pytest.raises(RuntimeError, match="identity"):
        restore(payload, cleanup)
    assert len(cleanup) == 1


@pytest.mark.parametrize("last", [
    {"python": "/opt/env/another/bin/python"},
    {"activate": "/opt/env/another/bin/activate"},
    {"name": "another"}, {"source": "failed"}, [], None, "not JSON",
])
def test_last_acknowledgement_is_authoritative(last):
    valid = reply(expected_identity())
    final = valid | last if isinstance(last, dict) else last
    payload = json.dumps(valid) + "\n" + (
        final if isinstance(final, str) else json.dumps(final)
    )
    cleanup = []
    with pytest.raises(RuntimeError, match="packed"):
        restore(payload, cleanup)
    assert len(cleanup) == 1


def test_complete_final_acknowledgement_allows_prior_logs_and_blank_lines():
    valid = reply(expected_identity())
    cleanup = []
    payload = "reader log\n" + json.dumps(valid) + "\n \n"
    assert restore(payload, cleanup) == valid["python"]
    assert len(cleanup) == 1
