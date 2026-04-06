"""Shared experiment status utilities.

Single source of truth for counting completed tasks, errors, and inferring
experiment status. Used by both check_status.py (CLI) and monitor.py (Gradio UI).

Key design decisions:
- Only reads top-level task dirs (not recursive into retry subdirs)
- Errors are per-unique-task (no double-counting retries)
- status.json is only trusted when completed >= total
"""

import json
import os
import re
from datetime import datetime as dt
from pathlib import Path

STALE_THRESHOLD_SEC = 10 * 60  # 10 minutes


def count_summary_infos(exp_dir: Path) -> int:
    """Count completed tasks — top-level task dirs with summary_info.json."""
    count = 0
    try:
        with os.scandir(exp_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith(("_", ".")):
                    si = os.path.join(entry.path, "summary_info.json")
                    if os.path.isfile(si):
                        count += 1
    except (FileNotFoundError, PermissionError):
        pass
    return count


def count_task_dirs(exp_dir: Path) -> int:
    """Count total task directories (excluding _ and . prefixed)."""
    try:
        return sum(
            1
            for d in os.scandir(exp_dir)
            if d.is_dir(follow_symlinks=False) and not d.name.startswith(("_", "."))
        )
    except (FileNotFoundError, PermissionError):
        return 0


def get_status_json(exp_dir: Path) -> str | None:
    """Read status from status.json if it exists."""
    status_file = exp_dir / "status.json"
    if status_file.exists():
        try:
            with open(status_file, "rb") as f:
                return json.loads(f.read()).get("status")
        except Exception:
            pass
    return None


def get_last_experiment_log_time(exp_dir: Path) -> float | None:
    """Return mtime of the most recent experiment.log in the tree, or None."""
    latest = None

    def walk(dirpath):
        nonlocal latest
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        if not entry.name.startswith("_"):
                            walk(entry.path)
                    elif entry.is_file() and entry.name == "experiment.log":
                        try:
                            mt = entry.stat().st_mtime
                            if latest is None or mt > latest:
                                latest = mt
                        except FileNotFoundError:
                            pass
        except (FileNotFoundError, PermissionError):
            pass

    walk(exp_dir)
    return latest


ERROR_THRESHOLD = 2  # experiments with > this many errors are flagged even if complete


def get_study_start_time(exp_dir: Path) -> float | None:
    """Return mtime of study.pkl.gz (marks when the study was created/massage started)."""
    study_pkl = exp_dir / "study.pkl.gz"
    if study_pkl.exists():
        return study_pkl.stat().st_mtime
    return None


def is_preparing(exp_dir: Path) -> bool:
    """Check if a .preparing marker exists (written by run_webarena.py during massage/reset)."""
    return (exp_dir / ".preparing").exists()


def infer_status(
    completed: int,
    total: int,
    status_json: str | None,
    last_log_mtime: float | None,
    error_count: int = 0,
    study_start_mtime: float | None = None,
    has_preparing_marker: bool = False,
) -> str:
    """Infer experiment status from counts and metadata.

    Returns one of: FINISHED, FINISHED_WITH_ERRORS, RUNNING, STALLED, PREPARING, EMPTY, UNKNOWN.

    PREPARING is returned when a study exists (study.pkl.gz) but no experiment.log
    has been written yet — the experiment is in the massage/reset phase.

    FINISHED_WITH_ERRORS is returned when all tasks completed but error_count > ERROR_THRESHOLD.
    This indicates the experiment needs attention (retry or manual override).

    Note: status.json is only trusted when completed >= total. A previous run
    may have written "finished" after exhausting retries, but a relaunch with -r
    adds new incomplete tasks.
    """
    # .preparing marker is the strongest signal (written by run_webarena.py)
    if has_preparing_marker:
        return "PREPARING"
    if total == 0:
        if study_start_mtime is not None:
            return "PREPARING"
        return "EMPTY"
    if completed >= total:
        if error_count > ERROR_THRESHOLD:
            return "FINISHED_WITH_ERRORS"
        return "FINISHED"
    # Only trust status.json when all tasks are actually complete.
    if status_json == "finished" and completed >= total:
        if error_count > ERROR_THRESHOLD:
            return "FINISHED_WITH_ERRORS"
        return "FINISHED"
    if last_log_mtime is not None:
        elapsed_sec = (dt.now() - dt.fromtimestamp(last_log_mtime)).total_seconds()
        # Check if study was relaunched AFTER the last log (massage/reset phase of relaunch)
        # Use a longer threshold for PREPARING (massage/reset can take 20+ min for VWA)
        PREPARING_THRESHOLD_SEC = 30 * 60  # 30 minutes
        if study_start_mtime is not None and study_start_mtime > last_log_mtime:
            study_elapsed = (dt.now() - dt.fromtimestamp(study_start_mtime)).total_seconds()
            if study_elapsed < PREPARING_THRESHOLD_SEC:
                return "PREPARING"
        if elapsed_sec > STALE_THRESHOLD_SEC:
            return "STALLED"
        return "RUNNING"
    # No experiment.log yet — check if study is preparing (massage/reset phase)
    if study_start_mtime is not None:
        return "PREPARING"
    return "UNKNOWN"


def load_avg_reward_and_errors(exp_dir: Path) -> tuple[float | None, int]:
    """Load average reward and error count from top-level task dirs only."""
    rewards = []
    errors = 0

    try:
        with os.scandir(exp_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith(("_", ".")):
                    si = os.path.join(entry.path, "summary_info.json")
                    if os.path.isfile(si):
                        try:
                            with open(si, "rb") as f:
                                data = json.loads(f.read())
                            rewards.append(data.get("cum_reward", 0))
                            if data.get("err_msg"):
                                errors += 1
                        except Exception:
                            pass
    except (FileNotFoundError, PermissionError):
        pass

    if not rewards:
        return None, 0

    return sum(rewards) / len(rewards), errors


def extract_task_id(task_subdir: str):
    """Extract task identifier from a task subdir name.

    WebArena/VWA: ..._on_webarena.130_0 -> 130
    MiniWoB:      ..._on_miniwob.click-button_14 -> miniwob.click-button_14
    WorkArena:    ..._on_workarena.servicenow.filter-list_0 -> workarena.servicenow.filter-list_0
    """
    m = re.search(r"_on_(.+)$", task_subdir)
    if m:
        full_task = m.group(1)
        parts = full_task.split(".")
        if len(parts) >= 2:
            tail = parts[-1]
            id_part, _, _ = tail.partition("_")
            try:
                return int(id_part)
            except ValueError:
                return full_task

    # Fallback: original logic
    _, _, right = task_subdir.rpartition(".")
    task_id, _, seed = right.partition("_")
    try:
        return int(task_id)
    except ValueError:
        return task_id


def get_experiment_info(exp_dir: Path) -> dict:
    """Get full status info for a single experiment. Returns JSON-serializable dict."""
    completed = count_summary_infos(exp_dir)
    total = count_task_dirs(exp_dir)
    status_json = get_status_json(exp_dir)
    last_log_mtime = get_last_experiment_log_time(exp_dir)
    study_start = get_study_start_time(exp_dir)
    avg_reward, error_count = load_avg_reward_and_errors(exp_dir)
    preparing_marker = is_preparing(exp_dir)
    status = infer_status(
        completed, total, status_json, last_log_mtime,
        error_count=error_count, study_start_mtime=study_start,
        has_preparing_marker=preparing_marker,
    )

    last_timestamp = None
    last_update_sec = None
    if last_log_mtime is not None:
        last_timestamp = dt.fromtimestamp(last_log_mtime).isoformat()
        last_update_sec = (dt.now() - dt.fromtimestamp(last_log_mtime)).total_seconds()

    preparing_sec = None
    if status == "PREPARING":
        # Use study.pkl.gz mtime as the start of preparation
        ref_time = study_start if study_start is not None else None
        if ref_time is not None:
            preparing_sec = (dt.now() - dt.fromtimestamp(ref_time)).total_seconds()

    return {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "completed": completed,
        "total": total,
        "status": status,
        "avg_reward": round(avg_reward, 4) if avg_reward is not None else None,
        "errors": error_count,
        "last_timestamp": last_timestamp,
        "last_update_sec": round(last_update_sec) if last_update_sec is not None else None,
        "preparing_sec": round(preparing_sec) if preparing_sec is not None else None,
    }


def scan_all_experiments(agentlab_dir: Path) -> list[dict]:
    """Scan all top-level experiments and return a list of status dicts."""
    results = []
    try:
        for entry in sorted(agentlab_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue
            # Skip organized subfolders (they contain experiments inside)
            if not re.match(r"\d{4}-\d{2}-\d{2}_", entry.name):
                continue
            info = get_experiment_info(entry)
            results.append(info)
    except (FileNotFoundError, PermissionError):
        pass
    return results


if __name__ == "__main__":
    """CLI: output JSON status for all experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment status (JSON output)")
    parser.add_argument("--dir", type=Path, default=None, help="agentlab_results directory")
    parser.add_argument("--experiment", type=str, default=None, help="Single experiment name or substring")
    args = parser.parse_args()

    if args.dir is None:
        # Auto-detect
        project_root = Path(__file__).resolve().parent.parent
        args.dir = project_root / "agentlab_results"

    if args.experiment:
        # Find matching experiment
        matches = [d for d in args.dir.iterdir() if d.is_dir() and args.experiment in d.name]
        results = [get_experiment_info(m) for m in matches]
    else:
        results = scan_all_experiments(args.dir)

    print(json.dumps(results, indent=2))
