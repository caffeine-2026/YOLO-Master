"""Repository-aware, path-safe helpers for the C3 Studio."""

from __future__ import annotations

import re
import socket
import subprocess
from pathlib import Path


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate the YOLO-Master repository root")


REPO_ROOT = _find_repo_root()
C3_ROOT = REPO_ROOT / "smoke" / "c3"
STUDIO_ROOT = C3_ROOT / "studio"
P0_ROOT = C3_ROOT / "p0"
P1_ROOT = C3_ROOT / "p1"
P2_ROOT = C3_ROOT / "p2"


def repo_path(relative: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve a repository-relative path without allowing traversal outside the repository."""
    candidate = (REPO_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository: {relative}") from exc
    if must_exist and not candidate.exists():
        raise FileNotFoundError(relative)
    return candidate


def relative_path(path: str | Path) -> str:
    """Return a POSIX repository-relative path safe for display in the UI."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Only repository paths may be displayed") from exc


def git_info() -> dict[str, str]:
    """Read a small fixed set of git metadata without accepting user-supplied shell input."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "commit_short": run("rev-parse", "--short=12", "HEAD"),
    }


def recorded_refs() -> dict[str, str]:
    """Read BASE_REF from the existing P0 report; later stage refs are absent unless explicitly recorded."""
    report = P0_ROOT / "docs" / "C3_P0_FINAL_REPORT.md"
    text = report.read_text(encoding="utf-8")
    match = re.search(r"`BASE_REF`:\s*`([0-9a-f]{40})`", text)
    return {
        "BASE_REF": match.group(1) if match else "Not recorded",
        "P0_REF": "Not recorded",
        "P1_REF": "Not recorded",
        "P2_REF": "Not recorded",
    }


def find_available_port(candidates: tuple[int, ...] = (7860, 7861, 7862)) -> int:
    """Return the first loopback-only available port from the allowed Studio range."""
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("Ports 7860, 7861, and 7862 are all occupied")
