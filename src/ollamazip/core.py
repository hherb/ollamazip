"""Pure API for ollamazip operations.

This module contains all the business logic as plain functions with no I/O to
stdout/stderr, so it can be driven by both the CLI and a GUI. Progress is
reported via an optional callback.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


ARCHIVE_EXTENSION = ".ollamazip"
ARCHIVE_VERSION = 1
DEFAULT_TAG = "latest"
DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"

_BUF_SIZE = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

# message: str, fraction: Optional[float] in [0, 1] (None = indeterminate)
ProgressCallback = Callable[[str, Optional[float]], None]


def _noop_progress(_message: str, _fraction: Optional[float]) -> None:
    pass


# ---------------------------------------------------------------------------
# Ollama directory helpers
# ---------------------------------------------------------------------------

_LINUX_SYSTEM_HOME = Path("/usr/share/ollama/.ollama/models")


def _user_ollama_home() -> Path:
    """Per-user Ollama models dir: ``~/.ollama/models`` on every supported OS."""
    if platform.system() == "Windows":
        return Path(os.environ.get("USERPROFILE", Path.home())) / ".ollama" / "models"
    return Path.home() / ".ollama" / "models"


def ollama_home_candidates() -> list[Path]:
    """Ordered list of candidate Ollama model directories to probe.

    Order, highest priority first:
      1. ``$OLLAMA_MODELS`` if set (explicit override always wins).
      2. The per-user dir (``~/.ollama/models``, or the Windows equivalent).
      3. Linux only: ``/usr/share/ollama/.ollama/models`` — the default for
         the official systemd-service install, where Ollama runs as a
         dedicated ``ollama`` system user.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        rp = Path(os.path.expanduser(os.path.expandvars(str(p))))
        if rp not in seen:
            seen.add(rp)
            candidates.append(rp)

    env = os.environ.get("OLLAMA_MODELS")
    if env:
        add(Path(env))

    add(_user_ollama_home())

    if platform.system() == "Linux":
        add(_LINUX_SYSTEM_HOME)

    return candidates


def _has_manifests(home: Path) -> bool:
    manifests = home / "manifests"
    if not manifests.is_dir():
        return False
    try:
        for entry in manifests.rglob("*"):
            if entry.is_file():
                return True
    except OSError:
        return False
    return False


def ollama_home() -> Path:
    """Return the Ollama models root directory.

    If ``$OLLAMA_MODELS`` is set, it wins unconditionally (hard override).
    Otherwise probes :func:`ollama_home_candidates` and returns the first
    one that actually contains manifests; if none are populated, falls back
    to the first candidate (so a fresh unpack lands in a sensible default).
    """
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env)))

    candidates = ollama_home_candidates()
    for home in candidates:
        if _has_manifests(home):
            return home
    return candidates[0]


def ensure_writable_home(home: Path) -> None:
    """Verify *home* is writable, creating it if needed.

    Raises ``PermissionError`` with actionable guidance otherwise. The
    error message is specialised for Ollama's Linux systemd-service path,
    which is owned by the ``ollama`` system user and not writable by
    default for regular users.
    """
    try:
        home.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(_permission_hint(home, str(exc))) from exc
    if not os.access(home, os.W_OK | os.X_OK):
        raise PermissionError(_permission_hint(home, "directory is not writable"))


def _permission_hint(home: Path, reason: str) -> str:
    base = f"Cannot write to Ollama models dir {home}: {reason}."
    if platform.system() == "Linux" and home == _LINUX_SYSTEM_HOME:
        return (
            f"{base}\n"
            f"\nThis is the default location for Ollama installed as a systemd "
            f"service; it is owned by the 'ollama' system user.\n"
            f"To modify models in the running Ollama service, pick one:\n"
            f"  - Re-run the same ollamazip command with sudo.\n"
            f"  - Grant your user write access:\n"
            f"        sudo usermod -aG ollama $USER  # then log out/in\n"
            f"        sudo chmod -R g+w {home}\n"
            f"  - Or move Ollama's storage to a user-owned dir and tell the\n"
            f"    service to use it:\n"
            f"        sudo systemctl edit ollama.service\n"
            f"        # add: Environment=\"OLLAMA_MODELS=/path/you/own\"\n"
            f"        sudo systemctl restart ollama\n"
            f"        export OLLAMA_MODELS=/path/you/own\n"
        )
    return (
        f"{base}\n"
        f"Set OLLAMA_MODELS to a writable directory, or fix permissions on "
        f"{home}."
    )


def parse_model_ref(model_ref: str) -> tuple[str, str, str, str]:
    """Parse a model reference into (registry, namespace, model, tag)."""
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


def manifest_path(home: Path, registry: str, namespace: str,
                  model: str, tag: str) -> Path:
    return home / "manifests" / registry / namespace / model / tag


def blob_path(home: Path, digest: str) -> Path:
    """Ollama stores blobs as blobs/<algo>-<hex>."""
    return home / "blobs" / digest.replace(":", "-")


def human_size(size_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PiB"


# ---------------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_blob(path: Path, expected_digest: str) -> bool:
    algo, expected_hex = expected_digest.split(":", 1)
    if algo != "sha256":
        return True  # unknown algo: cannot verify, assume ok
    return _sha256_file(path) == expected_hex


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def _detect_compression(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"\x1f\x8b":
        return "gzip"
    if magic == b"\x28\xb5\x2f\xfd":
        return "zstd"
    return "none"


def has_zstd() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return False


def _open_tar_read(filepath: Path) -> tarfile.TarFile:
    compression = _detect_compression(filepath)
    if compression == "zstd":
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        fh = open(filepath, "rb")
        reader = dctx.stream_reader(fh)
        return tarfile.open(fileobj=reader, mode="r|")
    if compression == "gzip":
        return tarfile.open(filepath, "r:gz")
    return tarfile.open(filepath, "r:")


def _add_bytes_to_tar(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tf.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# Model info dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    registry: str
    namespace: str
    model: str
    tag: str
    manifest_path: Path
    size_bytes: int
    modified_time: float

    @property
    def full_ref(self) -> str:
        return f"{self.registry}/{self.namespace}/{self.model}:{self.tag}"

    @property
    def short_ref(self) -> str:
        """A readable reference (omits defaults)."""
        if self.registry == DEFAULT_REGISTRY and self.namespace == DEFAULT_NAMESPACE:
            return f"{self.model}:{self.tag}"
        if self.registry == DEFAULT_REGISTRY:
            return f"{self.namespace}/{self.model}:{self.tag}"
        return self.full_ref


@dataclass
class ArchiveInfo:
    path: Path
    archive_bytes: int
    model: str = ""
    tag: str = ""
    registry: str = ""
    namespace: str = ""
    full_ref: str = ""
    created_at: str = ""
    compression: str = ""
    blob_count: int = 0
    uncompressed_bytes: int = 0
    source_platform: str = ""
    entries: list[tuple[str, int]] | None = None


# ---------------------------------------------------------------------------
# List local models
# ---------------------------------------------------------------------------

def list_local_models(home: Optional[Path] = None) -> list[ModelInfo]:
    """Enumerate all Ollama manifests under ``home/manifests``.

    Each manifest file represents one model:tag pair. Size is computed from
    the sum of unique blob sizes referenced by the manifest (config + layers).
    """
    home = home or ollama_home()
    manifests_root = home / "manifests"
    if not manifests_root.exists():
        return []

    models: list[ModelInfo] = []
    for path in manifests_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(manifests_root)
        parts = rel.parts
        if len(parts) < 4:
            continue  # expected registry/namespace/model/tag
        registry = parts[0]
        namespace = parts[1]
        model = "/".join(parts[2:-1])
        tag = parts[-1]

        size = 0
        try:
            data = json.loads(path.read_bytes())
            digests: list[str] = []
            cfg = (data.get("config") or {}).get("digest")
            if cfg:
                digests.append(cfg)
            for layer in (data.get("layers") or []):
                d = layer.get("digest")
                if d:
                    digests.append(d)
            seen: set[str] = set()
            for d in digests:
                if d in seen:
                    continue
                seen.add(d)
                bp = blob_path(home, d)
                if bp.exists():
                    size += bp.stat().st_size
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            # Skip malformed manifests but still list them with size=0
            pass

        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0

        models.append(ModelInfo(
            registry=registry,
            namespace=namespace,
            model=model,
            tag=tag,
            manifest_path=path,
            size_bytes=size,
            modified_time=mtime,
        ))

    models.sort(key=lambda m: (m.namespace, m.model, m.tag))
    return models


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

def _unique_digests(manifest: dict) -> list[str]:
    digests: list[str] = []
    cfg = (manifest.get("config") or {}).get("digest")
    if cfg:
        digests.append(cfg)
    for layer in (manifest.get("layers") or []):
        d = layer.get("digest")
        if d:
            digests.append(d)
    seen: set[str] = set()
    out: list[str] = []
    for d in digests:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def pack_model(
    model_ref: str,
    output: Optional[Path] = None,
    compress: str = "auto",
    progress: Optional[ProgressCallback] = None,
    home: Optional[Path] = None,
) -> Path:
    """Pack an Ollama model into a portable archive and return the output path.

    Raises FileNotFoundError if the manifest or any blob is missing.
    """
    progress = progress or _noop_progress
    home = home or ollama_home()
    registry, namespace, model, tag = parse_model_ref(model_ref)

    manifest_file = manifest_path(home, registry, namespace, model, tag)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_file}. "
            f"Is '{model_ref}' pulled in Ollama? Try: ollama pull {model_ref}"
        )

    manifest_data = manifest_file.read_bytes()
    manifest = json.loads(manifest_data)
    unique_digests = _unique_digests(manifest)

    total_size = len(manifest_data)
    blob_paths: dict[str, Path] = {}
    for digest in unique_digests:
        bp = blob_path(home, digest)
        if not bp.exists():
            raise FileNotFoundError(
                f"Blob not found: {bp}. The model may be corrupted. "
                f"Try: ollama pull {model_ref}"
            )
        blob_paths[digest] = bp
        total_size += bp.stat().st_size

    if compress == "auto":
        compress = "zstd" if has_zstd() else "gzip"

    if output:
        out_path = Path(output)
    else:
        safe_name = f"{model}-{tag}".replace("/", "_").replace(":", "_")
        out_path = Path(f"{safe_name}{ARCHIVE_EXTENSION}")

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

    progress(f"Packing {model}:{tag} ({len(unique_digests)} blobs, {human_size(total_size)})", 0.0)

    raw_fh = None
    compressed_fh = None
    if compress == "zstd":
        import zstandard
        cctx = zstandard.ZstdCompressor(level=3, threads=-1)
        raw_fh = open(out_path, "wb")
        compressed_fh = cctx.stream_writer(raw_fh)
        tf = tarfile.open(fileobj=compressed_fh, mode="w|")
    elif compress == "gzip":
        tf = tarfile.open(out_path, "w:gz", compresslevel=6)
    elif compress == "none":
        tf = tarfile.open(out_path, "w:")
    else:
        raise ValueError(f"Unknown compression: {compress}")

    try:
        _add_bytes_to_tar(tf, "metadata.json", metadata_bytes)
        _add_bytes_to_tar(tf, "manifest.json", manifest_data)
        bytes_done = len(metadata_bytes) + len(manifest_data)
        for i, digest in enumerate(unique_digests, 1):
            bp = blob_paths[digest]
            arc_name = f"blobs/{digest.replace(':', '-')}"
            size = bp.stat().st_size
            progress(
                f"[{i}/{len(unique_digests)}] {arc_name} ({human_size(size)})",
                bytes_done / total_size if total_size else None,
            )
            tf.add(str(bp), arcname=arc_name)
            bytes_done += size
        progress("Finalizing archive", 1.0)
    finally:
        tf.close()
        if compressed_fh is not None:
            compressed_fh.close()
        if raw_fh is not None:
            raw_fh.close()

    return out_path


# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------

def unpack_model(
    archive_path: Path,
    name: Optional[str] = None,
    verify: bool = False,
    progress: Optional[ProgressCallback] = None,
    home: Optional[Path] = None,
) -> ModelInfo:
    """Unpack an archive into the local Ollama store and return the installed ModelInfo."""
    progress = progress or _noop_progress
    home = home or ollama_home()
    ensure_writable_home(home)
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    progress(f"Reading archive: {archive_path.name}", None)
    tf = _open_tar_read(archive_path)

    metadata: Optional[dict] = None
    manifest_data: Optional[bytes] = None
    blobs_written = 0
    blobs_skipped = 0

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
                dest = home / "blobs" / blob_name
                if dest.exists() and dest.stat().st_size == member.size:
                    blobs_skipped += 1
                    # Still drain the stream for streaming tar
                    f = tf.extractfile(member)
                    if f:
                        while f.read(_BUF_SIZE):
                            pass
                    progress(f"Reusing {blob_name} ({human_size(member.size)})", None)
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
                    blobs_written += 1
                    progress(f"Extracted {blob_name} ({human_size(member.size)})", None)
    finally:
        tf.close()

    if manifest_data is None:
        raise ValueError("Archive contains no manifest.json")

    if metadata is None:
        metadata = {}

    if name:
        registry, namespace, model, tag = parse_model_ref(name)
    else:
        registry = metadata.get("registry", DEFAULT_REGISTRY)
        namespace = metadata.get("namespace", DEFAULT_NAMESPACE)
        model = metadata.get("model", "unknown")
        tag = metadata.get("tag", DEFAULT_TAG)

    manifest_dest = manifest_path(home, registry, namespace, model, tag)
    manifest_dest.parent.mkdir(parents=True, exist_ok=True)
    manifest_dest.write_bytes(manifest_data)

    progress(
        f"Installed {model}:{tag} ({blobs_written} new, {blobs_skipped} reused)",
        1.0,
    )

    if verify:
        manifest = json.loads(manifest_data)
        digests = _unique_digests(manifest)
        for i, digest in enumerate(digests, 1):
            bp = blob_path(home, digest)
            progress(f"Verifying [{i}/{len(digests)}] {digest}", i / len(digests))
            if not bp.exists():
                raise FileNotFoundError(f"Missing blob after unpack: {digest}")
            if not verify_blob(bp, digest):
                raise ValueError(f"Corrupt blob: {digest}")

    return ModelInfo(
        registry=registry,
        namespace=namespace,
        model=model,
        tag=tag,
        manifest_path=manifest_dest,
        size_bytes=sum(
            blob_path(home, d).stat().st_size
            for d in _unique_digests(json.loads(manifest_data))
            if blob_path(home, d).exists()
        ),
        modified_time=manifest_dest.stat().st_mtime,
    )


# ---------------------------------------------------------------------------
# Inspect archive
# ---------------------------------------------------------------------------

def inspect_archive(archive_path: Path) -> ArchiveInfo:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    info = ArchiveInfo(path=archive_path, archive_bytes=archive_path.stat().st_size)
    entries: list[tuple[str, int]] = []

    tf = _open_tar_read(archive_path)
    try:
        for member in tf:
            entries.append((member.name, member.size))
            if member.name == "metadata.json":
                f = tf.extractfile(member)
                if f:
                    meta = json.loads(f.read())
                    info.model = meta.get("model", "")
                    info.tag = meta.get("tag", "")
                    info.registry = meta.get("registry", "")
                    info.namespace = meta.get("namespace", "")
                    info.full_ref = meta.get("full_ref", "")
                    info.created_at = meta.get("created_at", "")
                    info.compression = meta.get("compression", "")
                    info.blob_count = meta.get("blob_count", 0)
                    info.uncompressed_bytes = meta.get("uncompressed_bytes", 0)
                    info.source_platform = meta.get("platform", "")
    finally:
        tf.close()

    info.entries = entries
    return info


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def _collect_referenced_blobs(home: Path) -> set[str]:
    """Return the set of blob filenames currently referenced by any manifest."""
    referenced: set[str] = set()
    manifests_root = home / "manifests"
    if not manifests_root.exists():
        return referenced
    for path in manifests_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        cfg = (data.get("config") or {}).get("digest")
        if cfg:
            referenced.add(cfg.replace(":", "-"))
        for layer in (data.get("layers") or []):
            d = layer.get("digest")
            if d:
                referenced.add(d.replace(":", "-"))
    return referenced


def delete_local_model(
    model_ref: str,
    prune_blobs: bool = True,
    home: Optional[Path] = None,
) -> tuple[int, int]:
    """Delete a local model's manifest and optionally prune orphaned blobs.

    Returns (manifests_removed, blobs_pruned).
    """
    home = home or ollama_home()
    ensure_writable_home(home)
    registry, namespace, model, tag = parse_model_ref(model_ref)
    target = manifest_path(home, registry, namespace, model, tag)
    if not target.exists():
        raise FileNotFoundError(f"Manifest not found: {target}")

    target.unlink()

    # Clean up empty parent directories up to (but not including) manifests/
    manifests_root = home / "manifests"
    parent = target.parent
    while parent != manifests_root and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent

    if not prune_blobs:
        return 1, 0

    referenced = _collect_referenced_blobs(home)
    blobs_dir = home / "blobs"
    removed = 0
    if blobs_dir.exists():
        for blob in blobs_dir.iterdir():
            if blob.is_file() and blob.name not in referenced:
                try:
                    blob.unlink()
                    removed += 1
                except OSError:
                    pass
    return 1, removed


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------

def rename_local_model(
    old_ref: str,
    new_ref: str,
    home: Optional[Path] = None,
) -> ModelInfo:
    """Rename a local model by copying its manifest to the new ref and deleting the old.

    Blobs are content-addressed and untouched.
    """
    home = home or ollama_home()
    ensure_writable_home(home)
    old_reg, old_ns, old_model, old_tag = parse_model_ref(old_ref)
    new_reg, new_ns, new_model, new_tag = parse_model_ref(new_ref)

    src = manifest_path(home, old_reg, old_ns, old_model, old_tag)
    dst = manifest_path(home, new_reg, new_ns, new_model, new_tag)

    if not src.exists():
        raise FileNotFoundError(f"Source manifest not found: {src}")
    if dst.exists() and src.resolve() != dst.resolve():
        raise FileExistsError(f"Destination already exists: {dst}")

    if src.resolve() == dst.resolve():
        # No-op rename
        return ModelInfo(
            registry=new_reg, namespace=new_ns, model=new_model, tag=new_tag,
            manifest_path=dst, size_bytes=0, modified_time=dst.stat().st_mtime,
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src.unlink()

    # Clean up empty source parents
    manifests_root = home / "manifests"
    parent = src.parent
    while parent != manifests_root and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent

    # Compute size for returned info
    try:
        manifest = json.loads(dst.read_bytes())
        size = 0
        for d in _unique_digests(manifest):
            bp = blob_path(home, d)
            if bp.exists():
                size += bp.stat().st_size
    except (OSError, json.JSONDecodeError, ValueError):
        size = 0

    return ModelInfo(
        registry=new_reg, namespace=new_ns, model=new_model, tag=new_tag,
        manifest_path=dst, size_bytes=size, modified_time=dst.stat().st_mtime,
    )
