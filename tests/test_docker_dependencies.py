import copy

from tools.docker_dependencies import requirements


def project():
    return {
        "build-system": {"requires": ["setuptools>=69"]},
        "project": {
            "dependencies": ["core>=1"],
            "optional-dependencies": {"face": ["face==1"], "neural": ["torch==2"]},
        },
    }


def test_unrelated_model_extra_does_not_change_main_dependency_bytes():
    before = project()
    after = copy.deepcopy(before)
    after["project"]["optional-dependencies"]["neural"].append("ultralytics==1")
    assert requirements(before, ["face"]) == requirements(after, ["face"])
    after["project"]["optional-dependencies"]["face"].append("cv==2")
    assert requirements(before, ["face"]) != requirements(after, ["face"])


def test_core_image_does_not_install_unselected_gpu_extras():
    assert requirements(project()) == "core>=1\nsetuptools>=69\n"
