"""
ollamazip — Bundle an Ollama model into a single portable archive, or unpack one.

Usage:
    ollamazip pack <model>[:<tag>] [-o output.ollamazip]
    ollamazip unpack <file.ollamazip> [--name <model>:<tag>]
    ollamazip list <file.ollamazip>

Archive format:
    A tar archive (optionally gzip/zstd compressed) containing:
        manifest.json          — the Ollama manifest (with model name/tag metadata)
        blobs/sha256-<hex>     — all referenced layer blobs
        metadata.json          — archive metadata (source model, creation date, etc.)

The default compression is zstd if the ``zstandard`` library is available, otherwise gzip.
Use --compress={zstd,gzip,none} to override.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVE_EXTENSION = ".ollamazip"
ARCHIVE_VERSION = 1
DEFAULT_TAG = "latest"
DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"

# Buffer size for hashing / copying (1 MiB)
_BUF_SIZE = 1 << 20


# ---------------------------------------------------------------------------
# Ollama directory helpers
# ---------------------------------------------------------------------------

def _ollama_home() -> Path:
    """Return the Ollama models root directory."""
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    system = platform.system()
    if system == "Darwin":
        return Path.home() / ".ollama" / "models"
    if system == "Linux":
        return Path.home() / ".ollama" / "models"
    if system == "Windows":
        return Path(os.environ.get("USERPROFILE", Path.home())) / ".ollama" / "models"
    return Path.home() / ".ollama" / "models"


def _parse_model_ref(model_ref: str) -> tuple[str, str, str, str]:
    """Parse a model reference into (registry, namespace, model, tag).

    Supports formats:
        model
        model:tag
        namespace/model:tag
        registry/namespace/model:tag
    """
    # Split off tag
    if ":" in model_ref:
        base, tag = model_ref.rsplit(":", 1)
    else:
        base, tag = model_ref, DEFAULT_TAG

    parts = base.strip("/").split("/")
    if len(parts) == 1:
        return DEFAULT_REGISTRY, DEFAULT_NAMESPACE, parts[0], tag
    if len(parts) == 2:
        return DEFAULT_REGISTRY, parts[0], parts[1], tag
    if len(parts) >= 3:
        return parts[0], parts[1], "/".join(parts[2:]), tag
    return DEFAULT_REGISTRY, DEFAULT_NAMESPACE, base, tag


def _manifest_path(ollama_home: Path, registry: str, namespace: str,
                   model: str, tag: str) -> Path:
    """Return the filesystem path to an Ollama manifest."""
    return ollama_home / "manifests" / registry / namespace / model / tag


def _blob_path(ollama_home: Path, digest: str) -> Path:
    """Return the filesystem path to an Ollama blob.

    Ollama stores blobs as blobs/<algo>-<hex> (e.g. blobs/sha256-abcd...).
    """
    return ollama_home / "blobs" / digest.replace(":", "-")


# ---------------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute sha256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _verify_blob(path: Path, expected_digest: str) -> bool:
    """Verify a blob file matches its expected sha256 digest."""
    # expected_digest is like "sha256:abcdef..."
    algo, expected_hex = expected_digest.split(":", 1)
    if algo != "sha256":
        print(f"  WARNING: unsupported digest algorithm '{algo}', skipping verification")
        return True
    actual_hex = _sha256_file(path)
    return actual_hex == expected_hex


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def _detect_compression(filepath: Path) -> str:
    """Detect compression type from file magic bytes."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"\x1f\x8b":
        return "gzip"
    # zstd magic: 0xFD2FB528
    if magic == b"\x28\xb5\x2f\xfd":
        return "zstd"
    # plain tar magic at offset 257: "ustar"
    # but we can just try opening as tar
    return "none"


def _has_zstd() -> bool:
    """Check if zstandard library is available."""
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return False


def _open_tar_read(filepath: Path) -> tarfile.TarFile:
    """Open a tar archive for reading, handling zstd if needed."""
    compression = _detect_compression(filepath)
    if compression == "zstd":
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        fh = open(filepath, "rb")
        reader = dctx.stream_reader(fh)
        # Wrap in a tarfile — we need a seekable wrapper for tarfile
        # Use streaming mode (no seek)
        return tarfile.open(fileobj=reader, mode="r|")
    if compression == "gzip":
        return tarfile.open(filepath, "r:gz")
    return tarfile.open(filepath, "r:")


# ---------------------------------------------------------------------------
# Human-readable sizes
# ---------------------------------------------------------------------------

def _human_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PiB"


# ---------------------------------------------------------------------------
# PACK
# ---------------------------------------------------------------------------

def cmd_pack(args: argparse.Namespace) -> int:
    """Pack an Ollama model into a portable archive."""
    registry, namespace, model, tag = _parse_model_ref(args.model)
    ollama_home = _ollama_home()

    manifest_file = _manifest_path(ollama_home, registry, namespace, model, tag)
    if not manifest_file.exists():
        print(f"ERROR: manifest not found: {manifest_file}", file=sys.stderr)
        print(f"Is '{args.model}' pulled in Ollama? Try: ollama pull {args.model}",
              file=sys.stderr)
        return 1

    manifest_data = manifest_file.read_bytes()
    manifest = json.loads(manifest_data)

    # Collect all layer digests
    layers: list[dict] = manifest.get("layers", [])
    config_digest = manifest.get("config", {}).get("digest")
    all_digests: list[str] = []
    if config_digest:
        all_digests.append(config_digest)
    for layer in layers:
        digest = layer.get("digest")
        if digest:
            all_digests.append(digest)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_digests: list[str] = []
    for d in all_digests:
        if d not in seen:
            seen.add(d)
            unique_digests.append(d)

    # Verify all blobs exist
    total_size = len(manifest_data)
    blob_paths: dict[str, Path] = {}
    for digest in unique_digests:
        bp = _blob_path(ollama_home, digest)
        if not bp.exists():
            print(f"ERROR: blob not found: {bp}", file=sys.stderr)
            print(f"The model may be corrupted. Try: ollama pull {args.model}",
                  file=sys.stderr)
            return 1
        blob_paths[digest] = bp
        total_size += bp.stat().st_size

    # Choose compression
    compress = args.compress
    if compress == "auto":
        compress = "zstd" if _has_zstd() else "gzip"

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        safe_name = f"{model}-{tag}".replace("/", "_").replace(":", "_")
        ext = ARCHIVE_EXTENSION
        out_path = Path(f"{safe_name}{ext}")

    print(f"Packing model: {registry}/{namespace}/{model}:{tag}")
    print(f"  Blobs: {len(unique_digests)}")
    print(f"  Uncompressed size: {_human_size(total_size)}")
    print(f"  Compression: {compress}")
    print(f"  Output: {out_path}")

    # Build metadata
    metadata = {
        "archive_version": ARCHIVE_VERSION,
        "model": model,
        "tag": tag,
        "registry": registry,
        "namespace": namespace,
        "full_ref": f"{registry}/{namespace}/{model}:{tag}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blob_count": len(unique_digests),
        "uncompressed_bytes": total_size,
        "compression": compress,
        "platform": platform.platform(),
    }
    metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")

    # Write archive
    if compress == "zstd":
        import zstandard
        cctx = zstandard.ZstdCompressor(level=3, threads=-1)
        raw_fh = open(out_path, "wb")
        compressed_fh = cctx.stream_writer(raw_fh)
        tf = tarfile.open(fileobj=compressed_fh, mode="w|")
    elif compress == "gzip":
        tf = tarfile.open(out_path, "w:gz", compresslevel=6)
        raw_fh = None
        compressed_fh = None
    else:
        tf = tarfile.open(out_path, "w:")
        raw_fh = None
        compressed_fh = None

    try:
        # Add metadata.json
        _add_bytes_to_tar(tf, "metadata.json", metadata_bytes)

        # Add manifest.json
        _add_bytes_to_tar(tf, "manifest.json", manifest_data)

        # Add blobs
        for i, digest in enumerate(unique_digests, 1):
            bp = blob_paths[digest]
            arc_name = f"blobs/{digest.replace(':', '-')}"
            size = bp.stat().st_size
            print(f"  [{i}/{len(unique_digests)}] {arc_name} ({_human_size(size)})")
            tf.add(str(bp), arcname=arc_name)
    finally:
        tf.close()
        if compressed_fh is not None:
            compressed_fh.close()
        if raw_fh is not None:
            raw_fh.close()

    archive_size = out_path.stat().st_size
    ratio = (archive_size / total_size * 100) if total_size > 0 else 100
    print(f"\nDone! Archive: {out_path} ({_human_size(archive_size)}, {ratio:.1f}% of original)")
    return 0


def _add_bytes_to_tar(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add an in-memory bytes object to a tar archive."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tf.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# UNPACK
# ---------------------------------------------------------------------------

def cmd_unpack(args: argparse.Namespace) -> int:
    """Unpack an .ollamazip archive into the local Ollama model store."""
    archive_path = Path(args.file)
    if not archive_path.exists():
        print(f"ERROR: file not found: {archive_path}", file=sys.stderr)
        return 1

    ollama_home = _ollama_home()

    # Read archive
    print(f"Reading archive: {archive_path}")
    tf = _open_tar_read(archive_path)

    metadata: Optional[dict] = None
    manifest_data: Optional[bytes] = None
    blobs_written: list[str] = []
    blobs_skipped: list[str] = []

    try:
        for member in tf:
            if member.name == "metadata.json":
                f = tf.extractfile(member)
                if f:
                    metadata = json.loads(f.read())

            elif member.name == "manifest.json":
                f = tf.extractfile(member)
                if f:
                    manifest_data = f.read()

            elif member.name.startswith("blobs/"):
                blob_name = os.path.basename(member.name)
                dest = ollama_home / "blobs" / blob_name
                if dest.exists() and dest.stat().st_size == member.size:
                    # Blob already present (content-addressed), skip
                    blobs_skipped.append(blob_name)
                    # Still need to advance the tar stream
                    f = tf.extractfile(member)
                    if f:
                        # Drain the stream for streaming tar
                        while f.read(_BUF_SIZE):
                            pass
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    f = tf.extractfile(member)
                    if f:
                        with open(dest, "wb") as out:
                            while True:
                                chunk = f.read(_BUF_SIZE)
                                if not chunk:
                                    break
                                out.write(chunk)
                    blobs_written.append(blob_name)
                    print(f"  Extracted: {blob_name} ({_human_size(member.size)})")
    finally:
        tf.close()

    if manifest_data is None:
        print("ERROR: archive contains no manifest.json", file=sys.stderr)
        return 1

    if metadata is None:
        print("WARNING: archive contains no metadata.json, using defaults")
        metadata = {}

    # Determine target model name
    if args.name:
        registry, namespace, model, tag = _parse_model_ref(args.name)
    else:
        registry = metadata.get("registry", DEFAULT_REGISTRY)
        namespace = metadata.get("namespace", DEFAULT_NAMESPACE)
        model = metadata.get("model", "unknown")
        tag = metadata.get("tag", DEFAULT_TAG)

    # Write manifest
    manifest_dest = _manifest_path(ollama_home, registry, namespace, model, tag)
    manifest_dest.parent.mkdir(parents=True, exist_ok=True)
    manifest_dest.write_bytes(manifest_data)

    print(f"\nModel unpacked: {model}:{tag}")
    print(f"  Blobs written: {len(blobs_written)}")
    print(f"  Blobs reused (already present): {len(blobs_skipped)}")
    print(f"  Manifest: {manifest_dest}")
    print(f"\nReady to use: ollama run {model}:{tag}")

    # Verify blob integrity if requested
    if args.verify:
        print("\nVerifying blob integrity...")
        manifest = json.loads(manifest_data)
        all_digests = []
        config_digest = manifest.get("config", {}).get("digest")
        if config_digest:
            all_digests.append(config_digest)
        for layer in manifest.get("layers", []):
            d = layer.get("digest")
            if d:
                all_digests.append(d)

        errors = 0
        for digest in all_digests:
            bp = _blob_path(ollama_home, digest)
            if not bp.exists():
                print(f"  MISSING: {digest}")
                errors += 1
            elif not _verify_blob(bp, digest):
                print(f"  CORRUPT: {digest}")
                errors += 1
            else:
                print(f"  OK: {digest}")

        if errors:
            print(f"\n{errors} blob(s) failed verification!", file=sys.stderr)
            return 1
        print("All blobs verified OK.")

    return 0


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    """List the contents of an .ollamazip archive."""
    archive_path = Path(args.file)
    if not archive_path.exists():
        print(f"ERROR: file not found: {archive_path}", file=sys.stderr)
        return 1

    tf = _open_tar_read(archive_path)
    metadata: Optional[dict] = None
    entries: list[tuple[str, int]] = []

    try:
        for member in tf:
            entries.append((member.name, member.size))
            if member.name == "metadata.json":
                f = tf.extractfile(member)
                if f:
                    metadata = json.loads(f.read())
    finally:
        tf.close()

    if metadata:
        print(f"Model: {metadata.get('full_ref', 'unknown')}")
        print(f"Created: {metadata.get('created_at', 'unknown')}")
        print(f"Compression: {metadata.get('compression', 'unknown')}")
        print(f"Blobs: {metadata.get('blob_count', '?')}")
        print(f"Uncompressed size: {_human_size(metadata.get('uncompressed_bytes', 0))}")
        print(f"Source platform: {metadata.get('platform', 'unknown')}")
        print()

    print(f"{'Size':>12}  Name")
    print(f"{'----':>12}  ----")
    total = 0
    for name, size in entries:
        print(f"{_human_size(size):>12}  {name}")
        total += size
    print(f"\n{len(entries)} entries, {_human_size(total)} uncompressed")

    archive_size = archive_path.stat().st_size
    print(f"Archive size: {_human_size(archive_size)}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="ollamazip",
        description="Bundle Ollama models into portable archives.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pack
    p_pack = subparsers.add_parser("pack", help="Pack a model into an archive")
    p_pack.add_argument("model", help="Model name (e.g. llama3:8b, mymodel:latest)")
    p_pack.add_argument("-o", "--output", help="Output file path")
    p_pack.add_argument(
        "--compress",
        choices=["auto", "zstd", "gzip", "none"],
        default="auto",
        help="Compression method (default: auto = zstd if available, else gzip)",
    )

    # unpack
    p_unpack = subparsers.add_parser("unpack", help="Unpack an archive into Ollama")
    p_unpack.add_argument("file", help="Path to .ollamazip file")
    p_unpack.add_argument("--name", help="Override model name:tag on import")
    p_unpack.add_argument(
        "--verify",
        action="store_true",
        help="Verify blob SHA256 integrity after unpacking",
    )

    # list
    p_list = subparsers.add_parser("list", help="List archive contents")
    p_list.add_argument("file", help="Path to .ollamazip file")

    args = parser.parse_args()

    if args.command == "pack":
        return cmd_pack(args)
    if args.command == "unpack":
        return cmd_unpack(args)
    if args.command == "list":
        return cmd_list(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
