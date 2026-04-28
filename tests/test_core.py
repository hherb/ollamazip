"""Unit tests for ollamazip.core directory discovery and permission preflight."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ollamazip import core


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Ensure tests never see the real $OLLAMA_MODELS, $HOME, or system path."""
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    # Redirect the Linux system-service path to a non-existent tmp dir so a
    # populated /usr/share/ollama on the test host can't leak in.
    monkeypatch.setattr(core, "_LINUX_SYSTEM_HOME", tmp_path / "fake-system")
    yield


def _populate(home: Path) -> None:
    """Create a manifest file inside home so _has_manifests returns True."""
    target = home / "manifests" / "registry.ollama.ai" / "library" / "test" / "latest"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")


# ---------------------------------------------------------------------------
# ollama_home_candidates / ollama_home
# ---------------------------------------------------------------------------

def test_candidates_default_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    cands = core.ollama_home_candidates()
    assert cands == [
        Path(os.environ["HOME"]) / ".ollama" / "models",
        core._LINUX_SYSTEM_HOME,
    ]


def test_candidates_default_macos(monkeypatch):
    monkeypatch.setattr(core.platform, "system", lambda: "Darwin")
    cands = core.ollama_home_candidates()
    assert cands == [Path(os.environ["HOME"]) / ".ollama" / "models"]


def test_candidates_default_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(core.platform, "system", lambda: "Windows")
    cands = core.ollama_home_candidates()
    assert cands == [Path(os.environ["USERPROFILE"]) / ".ollama" / "models"]


def test_candidates_env_first(monkeypatch, tmp_path):
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "custom"))
    cands = core.ollama_home_candidates()
    assert cands[0] == tmp_path / "custom"


def test_candidates_env_expands_tilde(monkeypatch, tmp_path):
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    monkeypatch.setenv("OLLAMA_MODELS", "~/myollama")
    cands = core.ollama_home_candidates()
    assert cands[0] == Path(os.environ["HOME"]) / "myollama"


def test_candidates_dedup(monkeypatch, tmp_path):
    """If $OLLAMA_MODELS points at the user-local default, list stays unique."""
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    user_home = Path(os.environ["HOME"]) / ".ollama" / "models"
    monkeypatch.setenv("OLLAMA_MODELS", str(user_home))
    cands = core.ollama_home_candidates()
    assert cands.count(user_home) == 1
    assert core._LINUX_SYSTEM_HOME in cands


def test_home_env_hard_override(monkeypatch, tmp_path):
    """OLLAMA_MODELS wins even if the path is empty/missing."""
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "scratch"))
    assert core.ollama_home() == tmp_path / "scratch"


def test_home_picks_populated_candidate(monkeypatch, tmp_path):
    """When env is unset, picks the first candidate with manifests."""
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    user_home = Path(os.environ["HOME"]) / ".ollama" / "models"
    sys_home = tmp_path / "fake-system"
    monkeypatch.setattr(core, "_LINUX_SYSTEM_HOME", sys_home)
    monkeypatch.setattr(
        core,
        "ollama_home_candidates",
        lambda: [user_home, sys_home],
    )
    _populate(sys_home)
    assert core.ollama_home() == sys_home


def test_home_falls_back_to_first_when_empty(monkeypatch, tmp_path):
    """When nothing is populated, returns the first candidate (user-local)."""
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    user_home = Path(os.environ["HOME"]) / ".ollama" / "models"
    assert core.ollama_home() == user_home


def test_home_prefers_user_when_both_populated(monkeypatch, tmp_path):
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    user_home = Path(os.environ["HOME"]) / ".ollama" / "models"
    sys_home = tmp_path / "fake-system"
    monkeypatch.setattr(core, "_LINUX_SYSTEM_HOME", sys_home)
    monkeypatch.setattr(
        core,
        "ollama_home_candidates",
        lambda: [user_home, sys_home],
    )
    _populate(user_home)
    _populate(sys_home)
    assert core.ollama_home() == user_home


# ---------------------------------------------------------------------------
# ensure_writable_home
# ---------------------------------------------------------------------------

def test_ensure_writable_creates_dir(tmp_path):
    home = tmp_path / "fresh"
    core.ensure_writable_home(home)
    assert home.is_dir()


def test_ensure_writable_passes_for_writable_dir(tmp_path):
    core.ensure_writable_home(tmp_path)


def test_ensure_writable_raises_with_hint_for_readonly(tmp_path):
    ro = tmp_path / "readonly"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        with pytest.raises(PermissionError) as excinfo:
            core.ensure_writable_home(ro)
        assert "Cannot write to Ollama models dir" in str(excinfo.value)
    finally:
        ro.chmod(stat.S_IRWXU)


def test_ensure_writable_linux_system_path_hint(monkeypatch, tmp_path):
    """The system-service path gets the long systemd-specific hint."""
    monkeypatch.setattr(core.platform, "system", lambda: "Linux")
    fake_sys = tmp_path / "sys"
    fake_sys.mkdir()
    fake_sys.chmod(0o555)
    monkeypatch.setattr(core, "_LINUX_SYSTEM_HOME", fake_sys)
    try:
        with pytest.raises(PermissionError) as excinfo:
            core.ensure_writable_home(fake_sys)
        msg = str(excinfo.value)
        assert "systemd" in msg
        assert "sudo" in msg
        assert "OLLAMA_MODELS" in msg
        # Operation-agnostic — should not assume "unpack"
        assert "unpack <archive>" not in msg
    finally:
        fake_sys.chmod(stat.S_IRWXU)
