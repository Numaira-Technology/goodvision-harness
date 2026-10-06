"""Manual Docker name fallback stays responsive and rechecks selections on Retry."""

from __future__ import annotations

import queue

import pytest

from tools import customer_update_gui as gui


class Widget:
    def __init__(self):
        self.value = ""
        self.disabled = False
        self.hidden = True
        self.options = {}

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

    def configure(self, **options):
        self.options.update(options)

    def state(self, values):
        self.disabled = "disabled" in values

    def pack(self, **options):
        self.hidden = False

    def pack_forget(self):
        self.hidden = True

    def start(self, *args):
        pass

    def stop(self):
        pass


class Root:
    def __init__(self):
        self.callbacks = []
        self.destroyed = False

    def after(self, interval, callback):
        self.callbacks.append((interval, callback))

    def destroy(self):
        self.destroyed = True


@pytest.fixture
def window(tmp_path, monkeypatch):
    update_window = gui.UpdateWindow.__new__(gui.UpdateWindow)
    update_window.root = Root()
    update_window.directory = tmp_path
    update_window.events = queue.Queue()
    update_window.busy = False
    update_window.projects = []
    update_window.manual_choices = {}
    update_window.manual_selection = None
    update_window.manual_mode = False
    update_window.manual_name = ""
    for name in (
        "status",
        "detail",
        "progress",
        "close_button",
        "action_button",
        "selection",
        "selection_frame",
        "manual_frame",
        "manual_input",
        "manual_entry",
        "manual_button",
    ):
        setattr(update_window, name, Widget())
    update_window.manual_controls = (
        update_window.manual_entry,
        update_window.manual_button,
    )
    update_window.workers = []

    class Thread:
        def __init__(self, *, target, daemon):
            assert daemon is False
            self.target = target

        def start(self):
            update_window.workers.append(self.target)

    monkeypatch.setattr(gui.threading, "Thread", Thread)
    monkeypatch.setattr(gui, "load_package", lambda directory: {"component": "harness"})
    return update_window


def match(name, identity):
    return {
        "label": name,
        "selection": {
            "component": "harness",
            "names": {"harness": name},
            "ids": {"harness": identity},
        },
    }


def run_worker(window):
    window.workers.pop(0)()


def test_empty_automatic_discovery_offers_name_entry_without_customer_paths(window):
    window.show_projects([])
    assert not window.manual_frame.hidden
    assert "container or image name" in window.detail.get().lower()
    assert window.action_button.options["command"] == window.check_package
    assert not window.workers


def test_empty_manual_name_does_not_start_worker(window, monkeypatch):
    monkeypatch.setattr(gui, "find_named_installations", lambda *args: pytest.fail())
    window.manual_input.set("  ")
    window.find_manual()
    assert not window.busy
    assert not window.workers
    assert "Enter a container or image name" in window.detail.get()


def test_enter_starts_one_lookup_and_keyboard_retry_is_ignored_while_busy(
    window,
    monkeypatch,
):
    lookups = []
    monkeypatch.setattr(
        gui, "find_named_installations", lambda *args: lookups.append(args) or []
    )
    window.manual_input.set("my-container")
    window.find_manual(object())
    assert window.busy
    assert len(window.workers) == 1
    window.find_manual(object())
    window.retry_manual()
    assert len(window.workers) == 1
    run_worker(window)
    window.poll_events()
    assert not window.busy
    window.find_manual(object())
    run_worker(window)
    assert lookups == [("harness", "my-container"), ("harness", "my-container")]


def test_single_verified_name_installs_from_worker_and_only_displays_queue_updates(
    window,
    monkeypatch,
):
    choice = match("legacy-camera", "verified-container-id")
    lookups = []
    installations = []

    def lookup(component, name):
        lookups.append((component, name))
        return [choice]

    def install(directory, **kwargs):
        installations.append((directory, kwargs))
        kwargs["progress"]("Restarting GoodVision...")

    monkeypatch.setattr(gui, "find_named_installations", lookup)
    monkeypatch.setattr(gui, "install", install)
    window.show_projects([])
    window.manual_input.set("  legacy-camera  ")
    window.find_manual()
    assert window.busy
    assert not lookups and not installations
    assert window.manual_entry.options["state"] == "disabled"
    window.close()
    assert not window.root.destroyed
    run_worker(window)
    assert lookups == [("harness", "legacy-camera")]
    assert not installations
    assert window.status.get() == "Looking for your GoodVision installation..."
    window.poll_events()
    assert window.manual_selection == choice["selection"]
    assert not installations
    assert "verified-container-id" not in window.status.get() + window.detail.get()
    run_worker(window)
    assert len(installations) == 1
    assert installations[0][0] == window.directory
    assert installations[0][1]["selection"] == choice["selection"]
    assert "project" not in installations[0][1]
    assert window.status.get() == "Updating GoodVision..."
    window.poll_events()
    assert not window.busy
    assert window.manual_frame.hidden
    assert window.status.get() == "Update complete. Open GoodVision to continue."


def test_image_with_multiple_matches_requires_explicit_container_selection(
    window,
    monkeypatch,
):
    choices = [match("camera-one", "id-one"), match("camera-two", "id-two")]
    selected = []
    monkeypatch.setattr(gui, "find_named_installations", lambda *args: choices)
    monkeypatch.setattr(
        window, "start_install", lambda **kwargs: selected.append(kwargs)
    )
    window.manual_input.set("goodvision-harness:legacy")
    window.find_manual()
    run_worker(window)
    window.poll_events()
    assert not selected
    assert window.action_button.disabled
    assert not window.manual_frame.hidden
    assert window.selection.options["values"] == ["camera-one", "camera-two"]
    window.selection.set("camera-two")
    window.selection_changed()
    assert not window.action_button.disabled
    window.start_selected()
    assert selected == [{"selection": choices[1]["selection"]}]


def test_verified_project_match_describes_all_lanes_without_extra_confirmation(window):
    selection = {
        "component": "harness",
        "project": "goodvision-harness",
        "ids": {"harness": "id-one", "harness-02": "id-two"},
    }
    window.show_manual_choices([{"label": "camera-one", "selection": selection}])
    assert window.manual_selection == selection
    assert window.busy
    assert len(window.workers) == 1
    assert "All GoodVision lanes" in window.detail.get()
    assert "id-one" not in window.detail.get()


@pytest.mark.parametrize(
    "selection,expected",
    [
        (match("camera", "id")["selection"], "Only this GoodVision container"),
        (
            {"component": "contour", "names": {"contour": "gateway", "engine": "gpu"}},
            "gateway and engine",
        ),
    ],
)
def test_manual_picker_explains_selected_scope(window, selection, expected):
    choices = [
        {"label": "first", "selection": selection},
        match("other", "other-id"),
    ]
    window.show_manual_choices(choices)
    window.selection.set("first")
    window.selection_changed()
    assert expected in window.detail.get()
    assert not window.action_button.disabled


def test_no_manual_match_keeps_entry_available_and_retry_resolves_again(
    window,
    monkeypatch,
):
    lookups = []
    monkeypatch.setattr(
        gui, "find_named_installations", lambda *args: lookups.append(args) or []
    )
    window.manual_input.set("old-container")
    window.find_manual()
    run_worker(window)
    window.poll_events()
    assert not window.busy
    assert not window.manual_frame.hidden
    assert window.manual_entry.options["state"] == "normal"
    assert "No matching" in window.status.get()
    assert window.action_button.options["command"] == window.retry_manual
    window.action_button.options["command"]()
    run_worker(window)
    assert lookups == [("harness", "old-container"), ("harness", "old-container")]


def test_manual_install_retry_reverifies_the_stored_name_and_discards_stale_ids(
    window,
    monkeypatch,
):
    old = match("my-camera", "old-container-id")
    fresh = match("my-camera", "new-container-id")
    lookups = []
    installs = []

    def lookup(component, name):
        lookups.append(name)
        return [old if len(lookups) == 1 else fresh]

    def install(directory, **kwargs):
        installs.append(kwargs["selection"])
        if len(installs) == 1:
            raise gui.CustomerError("Open GoodVision first, then try again.")

    monkeypatch.setattr(gui, "find_named_installations", lookup)
    monkeypatch.setattr(gui, "install", install)
    window.manual_input.set("my-camera")
    window.find_manual()
    run_worker(window)
    window.poll_events()
    run_worker(window)
    window.poll_events()
    assert window.manual_selection is None
    assert not window.manual_frame.hidden
    assert window.manual_input.get() == "my-camera"
    assert window.action_button.options["command"] == window.retry_manual
    window.manual_input.set("different-name-not-submitted")
    window.action_button.options["command"]()
    assert not window.manual_selection
    assert installs == [old["selection"]]
    run_worker(window)
    window.poll_events()
    run_worker(window)
    window.poll_events()
    assert lookups == ["my-camera", "my-camera"]
    assert installs == [old["selection"], fresh["selection"]]


def test_automatic_rediscovery_clears_saved_manual_selection(window, monkeypatch):
    selected = match("old-camera", "stale-id")["selection"]
    window.manual_selection = selected
    window.manual_choices = {"old-camera": selected}
    window.manual_mode = True
    window.manual_name = "old-camera"
    monkeypatch.setattr(gui, "find_projects", lambda component: [])
    window.check_package()
    assert window.manual_selection is None
    assert not window.manual_choices
    assert not window.manual_mode
    run_worker(window)
    window.poll_events()
    assert not window.manual_frame.hidden
    assert not window.workers


def test_unexpected_lookup_error_is_friendly_without_private_details(
    window,
    monkeypatch,
):
    def lookup(component, name):
        raise RuntimeError("PRIVATE_KEY=secret-value, container internal-id")

    monkeypatch.setattr(gui, "find_named_installations", lookup)
    window.manual_input.set("my-container")
    window.find_manual()
    run_worker(window)
    window.poll_events()
    assert not window.manual_frame.hidden
    assert not window.busy
    assert "secret-value" not in window.detail.get()
    assert "internal-id" not in window.detail.get()
    assert window.action_button.options["command"] == window.retry_manual


def test_duplicate_display_names_still_identify_the_chosen_match(window, monkeypatch):
    choices = [match("camera", "id-one"), match("camera", "id-two")]
    selected = []
    monkeypatch.setattr(
        window, "start_install", lambda **kwargs: selected.append(kwargs)
    )
    window.show_manual_choices(choices)
    assert window.selection.options["values"] == ["camera", "camera (2)"]
    window.selection.set("camera (2)")
    window.start_selected()
    assert selected == [{"selection": choices[1]["selection"]}]
