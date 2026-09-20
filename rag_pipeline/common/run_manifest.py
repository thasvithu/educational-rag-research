"""Local run artifacts, explicit failures, and environment/code provenance."""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .config import Config, validate_run_id, within
from .load_corpus import digest_json, file_hash


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def environment(root: Path) -> dict:
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    code_files = sorted((root / "rag_pipeline").rglob("*.py"))
    code_files += sorted((root / "rag_pipeline").rglob("*.json"))
    code_files += [root / "02_data_cleaning" / n for n in ("clean_pdfs.py", "prepare_texts.py", "validate_corpus.py")]
    code_files += sorted(root.glob("requirements*.txt"))
    code_files += [root / "pyproject.toml"]
    status = git(root, "status", "--porcelain")
    try:
        poppler = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=True).stderr.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        poppler = "unavailable; required for PDF re-extraction, not prepared-text loading"
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "gpu": "not probed by this environment snapshot; see the model probe result when available",
            "poppler": poppler,
            "packages": dict(sorted(packages.items())), "git_commit": git(root, "rev-parse", "HEAD"),
            "git_branch": git(root, "branch", "--show-current"), "git_dirty": bool(status),
            "git_status": status.splitlines(),
            "code_sha256": {p.relative_to(root).as_posix(): file_hash(p) for p in code_files if p.is_file()}}


def create_run(root: Path, config: Config, run_id: str) -> tuple[Path, dict]:
    config.validate(root)
    validate_run_id(run_id)
    path = within(root, f"{config.output_root}/{run_id}")
    relative = (path / "stage_manifest.json").relative_to(root).as_posix()
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "-q", relative])
    if ignored.returncode != 0:
        raise ValueError("Run output must be ignored by Git")
    if git(root, "ls-files", "--", path.relative_to(root).as_posix()):
        raise ValueError("Run output contains tracked files")
    path.mkdir(parents=True, exist_ok=False)  # No overwrite or implicit resume.
    manifest = {"schema_version": 1, "stage": "phase1_inspection", "run_id": run_id,
                "status": "running", "started_at_utc": now(), "config_hash": digest_json(config.to_dict()),
                "seed": config.seed, "errors": []}
    write_json(path / "stage_manifest.json", manifest)
    return path, manifest


def finish_run(path: Path, manifest: dict, *, error: str | None = None) -> None:
    manifest.update(status="failed" if error else "complete", finished_at_utc=now(), errors=[error] if error else [])
    manifest["artifact_sha256"] = {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*"))
                                  if p.is_file() and p != path / "stage_manifest.json"}
    write_json(path / "stage_manifest.json", manifest)
