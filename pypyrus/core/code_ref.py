from __future__ import annotations

import subprocess


def collect_code_ref() -> str | None:
    """
    Return a best-effort Git code reference for the current workspace.

    Format:
        git:<commit_sha>:clean
        git:<commit_sha>:dirty

    Returns None when the current working directory is not inside a Git repo
    or Git metadata cannot be resolved.
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    if not commit:
        return None

    dirty_flag = "dirty" if status_output.strip() else "clean"
    return f"git:{commit}:{dirty_flag}"
