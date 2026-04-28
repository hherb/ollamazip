"""ollamazip command-line interface.

Thin wrapper around :mod:`ollamazip.core`. Business logic lives in core; this
module only handles argument parsing and terminal output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ollamazip import core


def _print_progress(message: str, fraction: Optional[float]) -> None:
    if fraction is None:
        print(f"  {message}")
    else:
        print(f"  [{fraction * 100:5.1f}%] {message}")


def cmd_pack(args: argparse.Namespace) -> int:
    try:
        out_path = core.pack_model(
            model_ref=args.model,
            output=Path(args.output) if args.output else None,
            compress=args.compress,
            progress=_print_progress,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    archive_size = out_path.stat().st_size
    print(f"\nDone! Archive: {out_path} ({core.human_size(archive_size)})")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    try:
        info = core.unpack_model(
            archive_path=Path(args.file),
            name=args.name,
            verify=args.verify,
            progress=_print_progress,
        )
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nModel unpacked: {info.short_ref}")
    print(f"  Manifest: {info.manifest_path}")
    print(f"\nReady to use: ollama run {info.short_ref}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    try:
        info = core.inspect_archive(Path(args.file))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if info.full_ref:
        print(f"Model: {info.full_ref}")
        print(f"Created: {info.created_at}")
        print(f"Compression: {info.compression}")
        print(f"Blobs: {info.blob_count}")
        print(f"Uncompressed size: {core.human_size(info.uncompressed_bytes)}")
        print(f"Source platform: {info.source_platform}")
        print()

    print(f"{'Size':>12}  Name")
    print(f"{'----':>12}  ----")
    total = 0
    for name, size in info.entries or []:
        print(f"{core.human_size(size):>12}  {name}")
        total += size
    print(f"\n{len(info.entries or [])} entries, {core.human_size(total)} uncompressed")
    print(f"Archive size: {core.human_size(info.archive_bytes)}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List locally installed Ollama models."""
    home = core.ollama_home()
    models = core.list_local_models(home=home)
    if not models:
        print("No local Ollama models found.")
        print("Searched:")
        for p in core.ollama_home_candidates():
            print(f"  - {p}")
        return 0

    print(f"{'Size':>12}  Model")
    print(f"{'----':>12}  -----")
    total = 0
    for m in models:
        print(f"{core.human_size(m.size_bytes):>12}  {m.short_ref}")
        total += m.size_bytes
    print(f"\n{len(models)} models, {core.human_size(total)} total (from {home})")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from ollamazip import gui
    except ImportError as e:
        print(
            "ERROR: GUI dependencies not installed.\n"
            "Install with: pip install ollamazip[gui]\n"
            f"(import error: {e})",
            file=sys.stderr,
        )
        return 1
    gui.run(host=args.host, port=args.port, native=args.native, show=not args.no_browser)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ollamazip",
        description="Bundle Ollama models into portable archives.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pack = subparsers.add_parser("pack", help="Pack a model into an archive")
    p_pack.add_argument("model", help="Model name (e.g. llama3:8b, mymodel:latest)")
    p_pack.add_argument("-o", "--output", help="Output file path")
    p_pack.add_argument(
        "--compress",
        choices=["auto", "zstd", "gzip", "none"],
        default="auto",
        help="Compression method (default: auto = zstd if available, else gzip)",
    )

    p_unpack = subparsers.add_parser("unpack", help="Unpack an archive into Ollama")
    p_unpack.add_argument("file", help="Path to .ollamazip file")
    p_unpack.add_argument("--name", help="Override model name:tag on import")
    p_unpack.add_argument(
        "--verify", action="store_true",
        help="Verify blob SHA256 integrity after unpacking",
    )

    p_list = subparsers.add_parser("list", help="List archive contents")
    p_list.add_argument("file", help="Path to .ollamazip file")

    subparsers.add_parser("models", help="List locally installed Ollama models")

    p_gui = subparsers.add_parser("gui", help="Launch the graphical interface")
    p_gui.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p_gui.add_argument("--port", type=int, default=8734, help="Port to bind (default: 8734)")
    p_gui.add_argument("--native", action="store_true",
                       help="Open in a native window (requires pywebview)")
    p_gui.add_argument("--no-browser", action="store_true",
                       help="Do not open a browser automatically")

    args = parser.parse_args()

    handlers = {
        "pack": cmd_pack,
        "unpack": cmd_unpack,
        "list": cmd_list,
        "models": cmd_models,
        "gui": cmd_gui,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
