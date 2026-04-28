"""NiceGUI front-end for ollamazip.

Launch with ``ollamazip gui``. Provides two tabs:
  * Local Models — list / pack / rename / delete installed Ollama models
  * Archives — browse a folder of .ollamazip files, inspect / unpack / move / delete

Requires the ``[gui]`` extra: ``pip install ollamazip[gui]``.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from nicegui import run as ng_run, ui

from ollamazip import core


# Default folder for archives — user-tweakable via the UI
DEFAULT_ARCHIVE_DIR = Path.home() / "Downloads"


# ---------------------------------------------------------------------------
# Shared progress state (thread-safe dictionary, polled by UI timer)
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.message: str = ""
        self.fraction: Optional[float] = None
        self.dirty: bool = False

    def update(self, message: str, fraction: Optional[float]) -> None:
        with self._lock:
            self.message = message
            self.fraction = fraction
            self.dirty = True

    def snapshot(self) -> tuple[bool, str, Optional[float]]:
        with self._lock:
            dirty = self.dirty
            self.dirty = False
            return dirty, self.message, self.fraction


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _expand(p: str) -> Path:
    """Expand ~ and environment variables in a path string."""
    return Path(p).expanduser()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class _State:
    """Process-wide UI state."""

    def __init__(self) -> None:
        self.archive_dir: Path = DEFAULT_ARCHIVE_DIR
        self.selected_model: Optional[core.ModelInfo] = None
        self.selected_archive: Optional[Path] = None
        self.busy: bool = False


def _build(state: _State) -> None:
    progress = _Progress()

    # --------------------------------------------------------------- progress
    with ui.dialog() as progress_dialog, ui.card().classes("w-96"):
        ui.label("Working...").classes("text-lg font-bold")
        progress_label = ui.label("").classes("text-sm text-gray-600")
        progress_bar = ui.linear_progress(value=0, show_value=False).props("instant-feedback")
    progress_dialog.props("persistent")

    def _poll_progress() -> None:
        dirty, msg, frac = progress.snapshot()
        if not dirty:
            return
        progress_label.text = msg
        if frac is None:
            progress_bar.props("indeterminate")
        else:
            progress_bar.props(remove="indeterminate")
            progress_bar.value = frac

    ui.timer(0.1, _poll_progress)

    # ------------------------------------------------------------------- nav
    with ui.header().classes("items-center"):
        ui.label("ollamazip").classes("text-xl font-bold")
        ui.space()
        home_label = ui.label(f"Ollama home: {core.ollama_home()}").classes("text-xs opacity-70")
        candidates = "\n".join(str(p) for p in core.ollama_home_candidates())
        home_label.tooltip(f"Searched (in order):\n{candidates}")

    with ui.tabs().classes("w-full") as tabs:
        tab_models = ui.tab("Local models", icon="storage")
        tab_archives = ui.tab("Archives", icon="archive")

    with ui.tab_panels(tabs, value=tab_models).classes("w-full"):
        with ui.tab_panel(tab_models):
            _models_tab(state, progress, progress_dialog)
        with ui.tab_panel(tab_archives):
            _archives_tab(state, progress, progress_dialog)


# ---------------------------------------------------------------------------
# Local models tab
# ---------------------------------------------------------------------------

def _models_tab(state: _State, progress: _Progress, progress_dialog: ui.dialog) -> None:
    columns = [
        {"name": "short_ref", "label": "Model", "field": "short_ref",
         "align": "left", "sortable": True},
        {"name": "size", "label": "Size", "field": "size",
         "align": "right", "sortable": True},
        {"name": "registry", "label": "Registry", "field": "registry",
         "align": "left", "sortable": True},
    ]
    table = ui.table(columns=columns, rows=[], row_key="full_ref", selection="single") \
        .classes("w-full").style("height: calc(100vh - 260px)")
    table.props('flat bordered dense virtual-scroll')

    def refresh() -> None:
        models = core.list_local_models()
        table.rows = [
            {
                "full_ref": m.full_ref,
                "short_ref": m.short_ref,
                "size": core.human_size(m.size_bytes),
                "size_raw": m.size_bytes,
                "registry": m.registry,
            }
            for m in models
        ]
        table.update()
        count_label.text = f"{len(models)} models"

    def selected_ref() -> Optional[str]:
        sel = table.selected
        if not sel:
            return None
        return sel[0]["full_ref"]

    async def on_pack() -> None:
        ref = selected_ref()
        if not ref:
            ui.notify("Select a model first", type="warning")
            return
        if state.busy:
            return

        _, _, model, tag = core.parse_model_ref(ref)
        default_name = f"{model}-{tag}".replace("/", "_").replace(":", "_") \
            + core.ARCHIVE_EXTENSION

        with ui.dialog() as dlg, ui.card().classes("w-[520px]"):
            ui.label(f"Pack {ref}").classes("text-lg font-bold")
            folder_in = ui.input("Output folder", value=str(state.archive_dir)) \
                .classes("w-full")
            file_in = ui.input("Filename", value=default_name).classes("w-full")
            compress_in = ui.select(
                ["auto", "zstd", "gzip", "none"], value="auto", label="Compression",
            ).classes("w-full")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dlg.submit(None)).props("flat")
                ui.button(
                    "Pack",
                    on_click=lambda: dlg.submit({
                        "folder": folder_in.value,
                        "name": file_in.value,
                        "compress": compress_in.value,
                    }),
                ).props("color=primary")
        opts = await dlg
        if opts is None:
            return

        folder = _expand(opts["folder"])
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            ui.notify(f"Can't create folder: {e}", type="negative")
            return
        out_path = folder / opts["name"]
        state.archive_dir = folder

        await _run_with_progress(
            progress_dialog, progress,
            lambda: core.pack_model(
                ref, output=out_path, compress=opts["compress"],
                progress=progress.update,
            ),
            on_success=lambda result: ui.notify(
                f"Packed to {result}", type="positive", timeout=5000),
            on_error=lambda e: ui.notify(f"Pack failed: {e}", type="negative"),
            state=state,
        )

    async def on_rename() -> None:
        ref = selected_ref()
        if not ref:
            ui.notify("Select a model first", type="warning")
            return

        with ui.dialog() as dlg, ui.card():
            ui.label("Rename model").classes("text-lg font-bold")
            ui.label(f"Current: {ref}").classes("text-sm")
            new_name = ui.input("New name:tag").classes("w-80")
            _, _, m, t = core.parse_model_ref(ref)
            new_name.value = f"{m}:{t}"
            with ui.row():
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Rename",
                          on_click=lambda: dlg.submit(new_name.value)).props("color=primary")
        result = await dlg
        if not result:
            return
        try:
            info = core.rename_local_model(ref, result)
            ui.notify(f"Renamed to {info.short_ref}", type="positive")
            refresh()
        except (FileNotFoundError, FileExistsError, PermissionError) as e:
            ui.notify(f"Rename failed: {e}", type="negative", multi_line=True)

    async def on_delete() -> None:
        ref = selected_ref()
        if not ref:
            ui.notify("Select a model first", type="warning")
            return

        with ui.dialog() as dlg, ui.card():
            ui.label(f"Delete {ref}?").classes("text-lg font-bold")
            ui.label("This removes the manifest and prunes blobs no longer referenced.") \
                .classes("text-sm text-gray-600")
            prune = ui.checkbox("Prune orphaned blobs", value=True)
            with ui.row():
                ui.button("Cancel", on_click=lambda: dlg.submit(None)).props("flat")
                ui.button("Delete", on_click=lambda: dlg.submit(prune.value)) \
                    .props("color=negative")
        answer = await dlg
        if answer is None:
            return
        try:
            manifests, blobs = core.delete_local_model(ref, prune_blobs=answer)
            ui.notify(
                f"Deleted 1 manifest, {blobs} orphaned blob(s) pruned",
                type="positive",
            )
            refresh()
        except (FileNotFoundError, PermissionError) as e:
            ui.notify(f"Delete failed: {e}", type="negative", multi_line=True)

    with ui.row().classes(
        "w-full items-center q-gutter-sm sticky bottom-0 z-10 "
        "bg-white dark:bg-slate-900 "
        "border-t border-slate-200 dark:border-slate-700 py-2"
    ):
        ui.button("Refresh", icon="refresh", on_click=refresh).props("flat")
        ui.button("Pack to archive", icon="archive", on_click=on_pack) \
            .props("color=primary")
        ui.button("Rename", icon="edit", on_click=on_rename).props("flat")
        ui.button("Delete", icon="delete", on_click=on_delete).props("color=negative flat")
        ui.space()
        count_label = ui.label("0 models").classes("text-sm opacity-70")

    refresh()


# ---------------------------------------------------------------------------
# Archives tab
# ---------------------------------------------------------------------------

def _archives_tab(state: _State, progress: _Progress, progress_dialog: ui.dialog) -> None:
    columns = [
        {"name": "name", "label": "File", "field": "name",
         "align": "left", "sortable": True},
        {"name": "size", "label": "Size", "field": "size",
         "align": "right", "sortable": True},
    ]
    table = ui.table(columns=columns, rows=[], row_key="path", selection="single") \
        .classes("w-full").style("height: calc(100vh - 320px)")
    table.props('flat bordered dense virtual-scroll')

    def refresh() -> None:
        rows = []
        if state.archive_dir.exists():
            for p in sorted(state.archive_dir.glob(f"*{core.ARCHIVE_EXTENSION}")):
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                rows.append({
                    "path": str(p),
                    "name": p.name,
                    "size": core.human_size(size),
                    "size_raw": size,
                })
        table.rows = rows
        table.update()
        count_label.text = f"{len(rows)} archive(s) in {state.archive_dir}"

    def selected_path() -> Optional[Path]:
        sel = table.selected
        if not sel:
            return None
        return Path(sel[0]["path"])

    def on_open_in_finder() -> None:
        """Open the archives folder in the OS file manager."""
        import subprocess
        import sys
        path = state.archive_dir
        if not path.exists():
            ui.notify(f"Folder does not exist: {path}", type="warning")
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif sys.platform == "win32":
                subprocess.run(["explorer", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except OSError as e:
            ui.notify(f"Could not open folder: {e}", type="negative")

    async def on_inspect() -> None:
        path = selected_path()
        if not path:
            ui.notify("Select an archive first", type="warning")
            return
        try:
            info = await ng_run.io_bound(core.inspect_archive, path)
        except (FileNotFoundError, ValueError) as e:
            ui.notify(f"Inspect failed: {e}", type="negative")
            return

        with ui.dialog() as dlg, ui.card().classes("w-[600px]"):
            ui.label(path.name).classes("text-lg font-bold")
            ui.separator()
            with ui.grid(columns=2).classes("gap-x-4 gap-y-1 text-sm"):
                for label, value in [
                    ("Model", info.full_ref or "—"),
                    ("Created", info.created_at or "—"),
                    ("Compression", info.compression or "—"),
                    ("Blobs", str(info.blob_count)),
                    ("Uncompressed", core.human_size(info.uncompressed_bytes)),
                    ("Archive size", core.human_size(info.archive_bytes)),
                    ("Source platform", info.source_platform or "—"),
                ]:
                    ui.label(label).classes("font-semibold")
                    ui.label(value)
            ui.separator()
            ui.label("Entries").classes("font-semibold")
            for name, size in (info.entries or [])[:200]:
                ui.label(f"{core.human_size(size):>12}  {name}") \
                    .classes("font-mono text-xs")
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=dlg.close).props("flat")
        dlg.open()

    async def on_unpack() -> None:
        path = selected_path()
        if not path:
            ui.notify("Select an archive first", type="warning")
            return
        if state.busy:
            return

        with ui.dialog() as dlg, ui.card():
            ui.label(f"Unpack {path.name}").classes("text-lg font-bold")
            rename_input = ui.input("Rename to name:tag (optional)").classes("w-80")
            verify = ui.checkbox("Verify blob checksums", value=False)
            with ui.row():
                ui.button("Cancel", on_click=lambda: dlg.submit(None)).props("flat")
                ui.button(
                    "Unpack",
                    on_click=lambda: dlg.submit({"name": rename_input.value or None,
                                                 "verify": verify.value}),
                ).props("color=primary")
        opts = await dlg
        if opts is None:
            return

        await _run_with_progress(
            progress_dialog, progress,
            lambda: core.unpack_model(
                path, name=opts["name"], verify=opts["verify"],
                progress=progress.update,
            ),
            on_success=lambda info: ui.notify(
                f"Installed {info.short_ref}", type="positive", timeout=5000),
            on_error=lambda e: ui.notify(f"Unpack failed: {e}", type="negative"),
            state=state,
        )

    async def on_move() -> None:
        path = selected_path()
        if not path:
            ui.notify("Select an archive first", type="warning")
            return

        with ui.dialog() as dlg, ui.card().classes("w-[520px]"):
            ui.label(f"Move {path.name}").classes("text-lg font-bold")
            target_in = ui.input("Target folder", value=str(state.archive_dir)) \
                .classes("w-full")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dlg.submit(None)).props("flat")
                ui.button("Move", on_click=lambda: dlg.submit(target_in.value)) \
                    .props("color=primary")
        target_str = await dlg
        if not target_str:
            return

        target = _expand(target_str)
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            ui.notify(f"Can't create target: {e}", type="negative")
            return

        dest = target / path.name
        if dest.exists():
            ui.notify(f"Destination already exists: {dest}", type="negative")
            return
        try:
            shutil.move(str(path), str(dest))
            ui.notify(f"Moved to {dest}", type="positive")
            refresh()
        except OSError as e:
            ui.notify(f"Move failed: {e}", type="negative")

    async def on_delete() -> None:
        path = selected_path()
        if not path:
            ui.notify("Select an archive first", type="warning")
            return

        with ui.dialog() as dlg, ui.card():
            ui.label(f"Delete {path.name}?").classes("text-lg font-bold")
            ui.label(f"Size: {core.human_size(path.stat().st_size)}") \
                .classes("text-sm text-gray-600")
            with ui.row():
                ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat")
                ui.button("Delete", on_click=lambda: dlg.submit(True)) \
                    .props("color=negative")
        confirm = await dlg
        if not confirm:
            return
        try:
            path.unlink()
            ui.notify(f"Deleted {path.name}", type="positive")
            refresh()
        except OSError as e:
            ui.notify(f"Delete failed: {e}", type="negative")

    def _folder_changed() -> None:
        state.archive_dir = _expand(folder_input.value)
        refresh()

    with ui.row().classes("w-full items-end q-gutter-sm"):
        folder_input = ui.input("Archives folder", value=str(state.archive_dir)) \
            .classes("flex-grow")
        folder_input.on("blur", _folder_changed)
        folder_input.on("keydown.enter", _folder_changed)
        ui.button("Open in file manager", icon="folder_open",
                  on_click=on_open_in_finder).props("flat")
        ui.button("Refresh", icon="refresh", on_click=refresh).props("flat")

    with ui.row().classes(
        "w-full items-center q-gutter-sm sticky bottom-0 z-10 "
        "bg-white dark:bg-slate-900 "
        "border-t border-slate-200 dark:border-slate-700 py-2"
    ):
        ui.button("Inspect", icon="info", on_click=on_inspect).props("flat")
        ui.button("Unpack", icon="unarchive", on_click=on_unpack) \
            .props("color=primary")
        ui.button("Move", icon="drive_file_move", on_click=on_move).props("flat")
        ui.button("Delete", icon="delete", on_click=on_delete) \
            .props("color=negative flat")
        ui.space()
        count_label = ui.label("0 archive(s)").classes("text-sm opacity-70")

    refresh()


# ---------------------------------------------------------------------------
# Run helper: execute a blocking core operation in a thread with a progress dialog
# ---------------------------------------------------------------------------

async def _run_with_progress(
    dialog: ui.dialog,
    progress: _Progress,
    work: Callable[[], Any],
    *,
    on_success: Callable[[Any], None],
    on_error: Callable[[Exception], None],
    state: _State,
) -> None:
    if state.busy:
        return
    state.busy = True
    progress.update("Starting...", 0.0)
    dialog.open()
    try:
        result = await ng_run.io_bound(work)
        on_success(result)
    except Exception as e:  # noqa: BLE001 — surface any failure to the user
        on_error(e)
    finally:
        dialog.close()
        state.busy = False


# ---------------------------------------------------------------------------
# Entry point called by cli.cmd_gui
# ---------------------------------------------------------------------------

def run(*, host: str = "127.0.0.1", port: int = 8734,
        native: bool = False, show: bool = True) -> None:
    state = _State()

    @ui.page("/")
    def _index() -> None:
        _build(state)

    ui.run(
        host=host,
        port=port,
        title="ollamazip",
        native=native,
        show=show and not native,
        reload=False,
        dark=None,  # follow system
    )
