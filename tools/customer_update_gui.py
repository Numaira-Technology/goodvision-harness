"""Windowed customer entry point for an adjacent GoodVision update package.

The frozen Windows executable needs no command line or file selection. All
package inspection and Docker operations run outside Tk's main thread.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

if __package__:
    from .customer_update import (
        CustomerError,
        DISABLED_MESSAGE,
        find_named_installations,
        find_projects,
        install,
        load_package,
    )
else:
    from customer_update import (
        CustomerError,
        DISABLED_MESSAGE,
        find_named_installations,
        find_projects,
        install,
        load_package,
    )


def package_directory(override: Path | None = None) -> Path:
    """Use the executable's folder rather than PyInstaller's temporary folder."""
    if override is not None:
        return Path(override).resolve()
    entry = sys.executable if getattr(sys, "frozen", False) else __file__
    return Path(entry).resolve().parent


def friendly_error(error: Exception) -> str:
    """Keep technical output and unexpected exception details out of the UI."""
    if isinstance(error, CustomerError):
        return str(error)
    message = str(error).lower()
    if (
        "recovery did not become ready" in message
        or "could not finish or restore" in message
    ):
        return (
            "The update could not finish or restore the previous version. "
            "Keep this folder and contact support."
        )
    if "previous images were restored" in message:
        return (
            "The update could not finish, so your previous version was restored. "
            "Select Retry to try again. If it still fails, contact support."
        )
    if "restore contour admission" in message:
        return (
            "GoodVision needs attention before you start analysis. "
            "Keep this update folder and contact support."
        )
    if any(word in message for word in ("lock", "another update")):
        return (
            "Another GoodVision update may still be running. Wait for it to "
            "finish, then select Retry. If no update is running, contact support."
        )
    if any(
        word in message
        for word in ("recovery", "restor", "restart", "warmup", "readiness")
    ):
        return (
            "GoodVision could not finish restarting. Keep this update folder "
            "and Docker Desktop open, then select Retry. If GoodVision still "
            "will not start, contact support before using it."
        )
    if any(
        word in message
        for word in (
            "package",
            "manifest",
            "checksum",
            "archive",
            "images.tar",
            "sha256",
        )
    ):
        return (
            "This update folder is incomplete or damaged. Extract the complete "
            "update package into a new folder, then open Update.exe again."
        )
    if any(
        word in message
        for word in ("session", "analysis", "cleanup", "in flight", "in-flight")
    ):
        return (
            "GoodVision is still finishing an analysis. Stop analysis on every "
            "computer using this installation, wait for it to finish, then "
            "select Retry."
        )
    if any(
        word in message
        for word in ("configuration", "compose", "environment", "settings changed")
    ):
        return (
            "The installed GoodVision settings could not be confirmed. Keep "
            "this update folder and contact your GoodVision support contact."
        )
    if any(word in message for word in ("docker", "context", "linux containers")):
        return (
            "Start Docker Desktop and wait until it is ready, then select Retry. "
            "If the update still cannot start, contact your GoodVision support contact."
        )
    return (
        "The update could not finish. Keep this update folder and Docker Desktop "
        "open, then select Retry. If it still fails, contact your GoodVision "
        "support contact."
    )


class UpdateWindow:
    """Coordinate background work through a queue; only the UI thread uses Tk."""

    def __init__(self, root: tk.Tk, directory: Path):
        self.root = root
        self.directory = directory
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.projects: list[str] = []
        self.manual_choices: dict[str, dict] = {}
        self.manual_selection: dict | None = None
        self.manual_mode = False
        self.manual_name = ""
        self.root.title("GoodVision Update")
        self.root.geometry("620x510")
        self.root.minsize(540, 460)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=28)
        self.main_frame = frame
        frame.pack(fill="both", expand=True)
        self.heading_label = ttk.Label(
            frame, text="GoodVision update", font=("Segoe UI", 18, "bold")
        )
        self.heading_label.pack(anchor="w")
        self.status = tk.StringVar(value="Preparing your update...")
        self.status_label = ttk.Label(
            frame, textvariable=self.status, font=("Segoe UI", 12), wraplength=540
        )
        self.status_label.pack(anchor="w", fill="x", pady=(20, 10))
        self.detail = tk.StringVar(
            value="Please keep this window open until the update finishes."
        )
        self.detail_label = ttk.Label(
            frame, textvariable=self.detail, wraplength=540, justify="left"
        )
        self.detail_label.pack(anchor="w", fill="x", pady=(0, 16))

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 18))
        self.manual_frame = ttk.Frame(frame)
        self.manual_label = ttk.Label(self.manual_frame, text="Container or image name")
        self.manual_label.pack(anchor="w", pady=(0, 6))
        manual_row = ttk.Frame(self.manual_frame)
        manual_row.pack(fill="x")
        self.manual_input = tk.StringVar()
        self.manual_entry = ttk.Entry(manual_row, textvariable=self.manual_input)
        self.manual_entry.pack(side="left", fill="x", expand=True)
        self.manual_entry.bind("<Return>", self.find_manual)
        self.manual_button = ttk.Button(
            manual_row, text="Find installation", command=self.find_manual
        )
        self.manual_button.pack(side="right", padx=(10, 0))
        self.manual_controls = (self.manual_entry, self.manual_button)
        self.selection_frame = ttk.Frame(frame)
        self.selection_label = ttk.Label(
            self.selection_frame, text="GoodVision installation"
        )
        self.selection_label.pack(anchor="w", pady=(0, 6))
        self.selection = ttk.Combobox(self.selection_frame, state="readonly")
        self.selection.pack(fill="x")
        self.selection.bind("<<ComboboxSelected>>", self.selection_changed)

        buttons = ttk.Frame(frame)
        buttons.pack(side="bottom", fill="x", pady=(20, 0))
        self.close_button = ttk.Button(buttons, text="Close", command=self.close)
        self.close_button.pack(side="right")
        self.action_button = ttk.Button(
            buttons, text="Retry", command=self.check_package
        )
        self.action_button.pack(side="right", padx=(0, 10))
        self.action_button.state(["disabled"])
        self.root.bind("<Configure>", self.resize_text)
        self.poll_timer = self.root.after(80, self.poll_events)
        # Allow Windows to draw the window before package verification begins.
        self.discovery_timer = self.root.after(150, self.check_package)

    def resize_text(self, event):
        if event.widget is self.root:
            width = max(320, event.width - 56)
            self.status_label.configure(wraplength=width)
            self.detail_label.configure(wraplength=width)

    def set_busy(self, busy: bool):
        self.busy = busy
        self.close_button.state(["disabled"] if busy else ["!disabled"])
        self.action_button.state(["disabled"] if busy else ["!disabled"])
        self.selection.configure(state="disabled" if busy else "readonly")
        for control in getattr(self, "manual_controls", ()):
            control.configure(state="disabled" if busy else "normal")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def close(self):
        if self.busy:
            self.detail.set(
                "Please keep this window open while GoodVision finishes updating."
            )
            return
        self.root.destroy()

    def progress_message(self, message: str):
        self.events.put(("progress", message))

    def check_package(self):
        if self.busy:
            return
        self.projects = []
        self.manual_choices = {}
        self.manual_selection = None
        self.manual_mode = False
        self.manual_name = ""
        self.selection_frame.pack_forget()
        self.action_button.configure(text="Retry", command=self.check_package)
        self.status.set("Checking your update...")
        self.detail.set("Please keep this window open until the update finishes.")
        self.set_busy(True)

        def inspect_package():
            try:
                manifest = load_package(self.directory)
                projects = sorted(set(find_projects(manifest["component"])))
                self.events.put(("projects", projects))
            except Exception as error:
                self.events.put(("failed", friendly_error(error)))

        threading.Thread(target=inspect_package, daemon=False).start()

    def selection_changed(self, _event=None):
        selected = self.selection.get()
        if not self.busy and selected in getattr(self, "manual_choices", {}):
            self.detail.set(self.manual_scope(self.manual_choices[selected]))
        self.action_button.state(
            ["!disabled"]
            if not self.busy
            and (
                selected in self.projects
                or selected in getattr(self, "manual_choices", {})
            )
            else ["disabled"]
        )

    def start_selected(self):
        if self.busy:
            return
        selected = self.selection.get()
        if selected in self.projects:
            self.start_install(selected)
        elif selected in getattr(self, "manual_choices", {}):
            self.start_install(selection=self.manual_choices[selected])

    def start_install(
        self, project: str | None = None, *, selection: dict | None = None
    ):
        self.manual_selection = selection
        self.status.set("Updating GoodVision...")
        message = "Please keep this window open until the update finishes."
        if selection is not None:
            message = self.manual_scope(selection) + " " + message
        self.detail.set(message)
        self.set_busy(True)

        def run_install():
            try:
                if selection is None:
                    install(
                        self.directory, project=project, progress=self.progress_message
                    )
                else:
                    install(
                        self.directory,
                        selection=selection,
                        progress=self.progress_message,
                    )
                self.events.put(("complete", None))
            except Exception as error:
                self.events.put(("failed", friendly_error(error)))

        threading.Thread(target=run_install, daemon=False).start()

    @staticmethod
    def manual_scope(selection: dict) -> str:
        if selection.get("component") == "contour":
            return "The Contour gateway and engine will be updated together."
        if selection.get("project"):
            return "All GoodVision lanes in this installation will be updated."
        return "Only this GoodVision container will be updated."

    def show_projects(self, projects: list[str]):
        self.projects = projects
        self.manual_choices = {}
        self.manual_selection = None
        self.manual_mode = False
        if len(projects) == 1:
            self.show_manual_input(False)
            self.start_install(projects[0])
            return
        self.set_busy(False)
        if not projects:
            self.show_manual_input(True)
            self.status.set("No GoodVision installation found on this computer")
            self.detail.set(
                "Enter the container or image name shown in Docker Desktop, "
                "then select Find installation."
            )
            self.action_button.configure(text="Retry", command=self.check_package)
            return
        self.show_manual_input(False)
        self.status.set("Choose which GoodVision installation to update.")
        self.detail.set("Select an installation below, then select Update.")
        self.selection.configure(values=projects)
        self.selection.set("")
        self.selection_frame.pack(fill="x")
        self.action_button.configure(text="Update", command=self.start_selected)
        self.selection_changed()

    def show_manual_input(self, visible: bool):
        frame = getattr(self, "manual_frame", None)
        if frame is not None:
            if visible:
                frame.pack(fill="x", pady=(0, 16))
            else:
                frame.pack_forget()

    def find_manual(self, _event=None):
        if self.busy:
            return
        name = self.manual_input.get().strip()
        if not name:
            self.detail.set("Enter a container or image name from Docker Desktop.")
            return
        self.manual_mode = True
        self.manual_name = name
        self.manual_selection = None
        self.manual_choices = {}
        self.projects = []
        self.selection_frame.pack_forget()
        self.status.set("Looking for your GoodVision installation...")
        self.detail.set(
            "Please keep this window open while the installation is checked."
        )
        self.set_busy(True)

        def find_installation():
            try:
                manifest = load_package(self.directory)
                choices = find_named_installations(manifest["component"], name)
                self.events.put(("manual_choices", choices))
            except Exception as error:
                self.events.put(("failed", friendly_error(error)))

        threading.Thread(target=find_installation, daemon=False).start()

    def retry_manual(self):
        if self.busy:
            return
        self.manual_input.set(self.manual_name)
        self.find_manual()

    def show_manual_choices(self, choices: list[dict]):
        self.manual_choices = {}
        self.manual_selection = None
        self.manual_mode = True
        self.show_manual_input(True)
        if len(choices) == 1:
            self.start_install(selection=choices[0]["selection"])
            return
        self.set_busy(False)
        if not choices:
            self.status.set("No matching GoodVision installation found.")
            self.detail.set(
                "Make sure GoodVision is running, then check the name in Docker "
                "Desktop and try again."
            )
            self.action_button.configure(text="Retry", command=self.retry_manual)
            return
        for choice in choices:
            base = choice["label"]
            label = base
            number = 2
            while label in self.manual_choices:
                label = f"{base} ({number})"
                number += 1
            self.manual_choices[label] = choice["selection"]
        self.status.set("Choose which GoodVision installation to update.")
        self.detail.set("Select the installation below, then select Update.")
        self.selection.configure(values=list(self.manual_choices))
        self.selection.set("")
        self.selection_frame.pack(fill="x")
        self.action_button.configure(text="Update", command=self.start_selected)
        self.selection_changed()

    def poll_events(self):
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "progress":
                    self.status.set(value)
                elif event == "projects":
                    self.show_projects(value)
                elif event == "manual_choices":
                    self.show_manual_choices(value)
                elif event == "complete":
                    self.set_busy(False)
                    self.selection_frame.pack_forget()
                    self.show_manual_input(False)
                    self.status.set("Update complete. Open GoodVision to continue.")
                    self.detail.set("You can now close this window.")
                    self.action_button.pack_forget()
                elif event == "failed":
                    self.set_busy(False)
                    self.selection_frame.pack_forget()
                    self.manual_selection = None
                    self.manual_choices = {}
                    self.status.set("The update needs your attention.")
                    self.detail.set(value)
                    if value == DISABLED_MESSAGE:
                        self.status.set("Customer automatic updates are disabled.")
                        self.show_manual_input(False)
                        self.action_button.pack_forget()
                        continue
                    self.action_button.configure(
                        text="Retry",
                        command=self.retry_manual
                        if getattr(self, "manual_mode", False)
                        else self.check_package,
                    )
                    if getattr(self, "manual_mode", False):
                        self.show_manual_input(True)
        except queue.Empty:
            pass
        self.poll_timer = self.root.after(80, self.poll_events)


def widget_layout(widget, frame, *, minimum_width=None):
    """Report requested-size clipping through every ancestor of a hidden widget."""
    width, height = widget.winfo_width(), widget.winfo_height()
    required_width = widget.winfo_reqwidth() if minimum_width is None else minimum_width
    fits = width >= required_width and height >= widget.winfo_reqheight()
    visible = True
    x = y = 0
    current = widget
    while current is not frame:
        visible = visible and bool(current.winfo_manager())
        x += current.winfo_x()
        y += current.winfo_y()
        parent = current.master
        if parent is None:
            return {"present": False, "fits": False, "bounds": [x, y, width, height]}
        fits = fits and (
            x >= 0
            and y >= 0
            and x + width <= parent.winfo_width()
            and y + height <= parent.winfo_height()
        )
        current = parent
    return {
        "present": visible,
        "fits": visible and fits,
        "bounds": [x, y, width, height],
    }


def hidden_layout_checks(window):
    """Exercise actual Tk geometry in fixed hidden frames, never Docker or timers."""
    root = window.root
    root.after_cancel(window.poll_timer)
    root.after_cancel(window.discovery_timer)
    root.unbind("<Configure>")
    # A withdrawn native window may remain 1x1. Give its real content frame the
    # same fixed area it receives from pack(fill='both', expand=True) when shown.
    frame = window.main_frame
    frame.pack_forget()
    choices = [
        {"label": "Example camera one", "selection": {"component": "harness"}},
        {"label": "Example camera two", "selection": {"component": "harness"}},
    ]
    checks = []
    for width, height in ((620, 510), root.minsize()):
        root.geometry(f"{width}x{height}")
        frame.place(x=0, y=0, width=width, height=height)
        window.status_label.configure(wraplength=width - 56)
        window.detail_label.configure(wraplength=width - 56)
        for state in ("manual_entry", "multiple_matches"):
            if state == "manual_entry":
                window.selection_frame.pack_forget()
                window.show_projects([])
            else:
                window.show_manual_choices(choices)
            root.update_idletasks()
            names = [
                "heading_label",
                "status_label",
                "detail_label",
                "progress",
                "manual_label",
                "manual_entry",
                "manual_button",
                "action_button",
                "close_button",
            ]
            if state == "multiple_matches":
                names += ["selection_label", "selection"]
            controls = {
                name: widget_layout(
                    getattr(window, name),
                    frame,
                    minimum_width=120
                    if name in {"manual_entry", "selection"}
                    else None,
                )
                for name in names
            }
            checks.append(
                {
                    "state": state,
                    "size": [width, height],
                    "controls": controls,
                    "fits": all(control["fits"] for control in controls.values()),
                }
            )
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--self-test-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = tk.Tk()
    if args.self_test_output:
        root.withdraw()
    window = UpdateWindow(root, package_directory(args.package_dir))
    if args.self_test_output:
        # Exercise the actual frozen Tk/Tcl runtime without opening a customer
        # window or processing the scheduled discovery/update callbacks.
        initial_status = window.status.get()
        layouts = hidden_layout_checks(window)
        fits = all(layout["fits"] for layout in layouts)
        args.self_test_output.write_text(
            json.dumps(
                {
                    "frozen": bool(getattr(sys, "frozen", False)),
                    "title": root.title(),
                    "status": initial_status,
                    "layout_ok": fits,
                    "layout_checks": layouts,
                }
            ),
            encoding="utf-8",
        )
        root.destroy()
        return 0 if fits else 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
