from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": safe_run(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "short_commit": safe_run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root),
        "branch": safe_run(["git", "branch", "--show-current"], cwd=repo_root),
        "remote_origin": safe_run(["git", "config", "--get", "remote.origin.url"], cwd=repo_root),
        "status_porcelain": safe_run(["git", "status", "--short"], cwd=repo_root),
    }


def package_file(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__file__", None)


def package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def python_snapshot() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uname": list(platform.uname()),
        "cwd": os.getcwd(),
    }


def torch_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    snapshot: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        snapshot["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return snapshot


def build_run_manifest(
    *,
    repo_root: Path,
    run_kind: str,
    cli_args: dict[str, Any],
    requested_datasets: list[str],
    evaluated_datasets: list[str],
    skipped_datasets: list[dict[str, str]],
    output_files: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "generated_at_utc": now_utc_iso(),
        "run_kind": run_kind,
        "command": " ".join(shlex_quote(arg) for arg in sys.argv),
        "repo_root": str(repo_root),
        "git": git_snapshot(repo_root),
        "python": python_snapshot(),
        "torch": torch_snapshot(),
        "laion_clap": {
            "version": package_version("laion_clap"),
            "module_file": package_file("laion_clap"),
        },
        "cli_args": cli_args,
        "requested_datasets": requested_datasets,
        "evaluated_datasets": evaluated_datasets,
        "skipped_datasets": skipped_datasets,
        "output_files": output_files,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def shlex_quote(token: str) -> str:
    if token == "":
        return "''"
    if all(char.isalnum() or char in "._/-=:" for char in token):
        return token
    return "'" + token.replace("'", "'\"'\"'") + "'"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
