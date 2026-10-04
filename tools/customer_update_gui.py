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
    from .customer_update import CustomerError, find_projects, install, load_package
else:
    from customer_update import CustomerError, find_projects, install, load_package


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
        self.root.title("GoodVision Update")
        self.root.geometry("620x390")
        self.root.minsize(540, 360)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=28)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="GoodVision update", font=("Segoe UI", 18, "bold")).pack(
            anchor="w"
        )
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
        self.selection_frame = ttk.Frame(frame)
        ttk.Label(self.selection_frame, text="GoodVision installation").pack(
            anchor="w", pady=(0, 6)
        )
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
        self.root.after(80, self.poll_events)
        # Allow Windows to draw the window before package verification begins.
        self.root.after(150, self.check_package)

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
        self.action_button.state(
            ["!disabled"]
            if self.selection.get() in self.projects and not self.busy
            else ["disabled"]
        )

    def start_selected(self):
        if not self.busy and self.selection.get() in self.projects:
            self.start_install(self.selection.get())

    def start_install(self, project: str):
        self.status.set("Updating GoodVision...")
        self.detail.set("Please keep this window open until the update finishes.")
        self.set_busy(True)

        def run_install():
            try:
                install(self.directory, project=project, progress=self.progress_message)
                self.events.put(("complete", None))
            except Exception as error:
                self.events.put(("failed", friendly_error(error)))

        threading.Thread(target=run_install, daemon=False).start()

    def show_projects(self, projects: list[str]):
        self.projects = projects
        if len(projects) == 1:
            self.start_install(projects[0])
            return
        self.set_busy(False)
        if not projects:
            self.status.set("No GoodVision installation found on this computer")
            self.detail.set(
                "Start Docker Desktop and make sure GoodVision is installed "
                "on this computer, then select Retry."
            )
            self.action_button.configure(text="Retry", command=self.check_package)
            return
        self.status.set("Choose which GoodVision installation to update.")
        self.detail.set("Select an installation below, then select Update.")
        self.selection.configure(values=projects)
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
                elif event == "complete":
                    self.set_busy(False)
                    self.selection_frame.pack_forget()
                    self.status.set("Update complete. Open GoodVision to continue.")
                    self.detail.set("You can now close this window.")
                    self.action_button.pack_forget()
                elif event == "failed":
                    self.set_busy(False)
                    self.selection_frame.pack_forget()
                    self.status.set("The update needs your attention.")
                    self.detail.set(value)
                    self.action_button.configure(
                        text="Retry", command=self.check_package
                    )
        except queue.Empty:
            pass
        self.root.after(80, self.poll_events)


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
        root.update_idletasks()
        args.self_test_output.write_text(
            json.dumps(
                {
                    "frozen": bool(getattr(sys, "frozen", False)),
                    "title": root.title(),
                    "status": window.status.get(),
                }
            ),
            encoding="utf-8",
        )
        root.destroy()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
