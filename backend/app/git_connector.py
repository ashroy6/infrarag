from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOS_DIR = Path(os.getenv("REPOS_DIR", "/app/data/repos"))


def ensure_repos_dir() -> Path:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    return REPOS_DIR


def clone_or_pull_repo(repo_url: str, repo_name: str | None = None, branch: str | None = None) -> str:
    ensure_repos_dir()

    if repo_name:
        target_dir = REPOS_DIR / repo_name
    else:
        name = repo_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        target_dir = REPOS_DIR / name

    if target_dir.exists():
        subprocess.run(["git", "-C", str(target_dir), "pull"], check=True)
    else:
        cmd = ["git", "clone", repo_url, str(target_dir)]
        if branch:
            cmd = ["git", "clone", "--branch", branch, repo_url, str(target_dir)]
        subprocess.run(cmd, check=True)

    return str(target_dir)
