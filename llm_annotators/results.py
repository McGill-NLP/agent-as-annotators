"""Reusable functions for loading experiment results and site/category mappings.

Extracted from apps/webarena_results_dashboard.py for use by both the dashboard
and standalone scripts (e.g., export_per_category_results.py).
"""
import importlib.resources
import os
import re
from collections import defaultdict
from pathlib import Path

import orjson


# ═══════════════════════════════════════════════════════════════
# Site/category mapping functions
# ═══════════════════════════════════════════════════════════════

WEBARENA_CONFIG = Path(__file__).resolve().parents[1] / "vars" / "wa_test_config_base.json"


def load_site_mapping_webarena() -> dict[int, str]:
    """Load task_id -> site mapping from WebArena test config."""
    with open(WEBARENA_CONFIG, "rb") as f:
        tasks = orjson.loads(f.read())
    mapping = {}
    for task in tasks:
        task_id = task["task_id"]
        sites = task["sites"]
        mapping[task_id] = sites[0] if len(sites) == 1 else "cross-site"
    return mapping


def load_site_mapping_webarena_all_splits() -> dict[int, str]:
    """Load task_id -> site mapping for ALL WebArena tasks (train + test)."""
    from browsergym.experiments.benchmark.metadata.utils import task_metadata

    tm = task_metadata("webarena")
    mapping = {}
    for _, row in tm.iterrows():
        task_id = int(row["task_name"].split(".")[-1])
        sites = row["sites"]
        if isinstance(sites, str):
            parts = sites.strip().split()
            mapping[task_id] = parts[0] if len(parts) == 1 else "cross-site"
        elif isinstance(sites, list):
            mapping[task_id] = sites[0] if len(sites) == 1 else "cross-site"
        else:
            mapping[task_id] = "unknown"
    return mapping


def load_site_mapping_visualwebarena() -> dict[int, str]:
    """Load task_id -> site mapping from VisualWebArena test config."""
    import visualwebarena

    raw = (
        importlib.resources.files(visualwebarena)
        .joinpath("test_raw.json")
        .read_text()
    )
    tasks = orjson.loads(raw)
    mapping = {}
    for task in tasks:
        task_id = task["task_id"]
        sites = task["sites"]
        mapping[task_id] = sites[0] if len(sites) == 1 else "cross-site"
    return mapping


def load_site_mapping_visualwebarena_all_splits() -> dict[int, str]:
    """Load task_id -> site mapping for ALL VisualWebArena tasks (train + test)."""
    from browsergym.experiments.benchmark.metadata.utils import task_metadata

    tm = task_metadata("visualwebarena")
    mapping = {}
    for _, row in tm.iterrows():
        task_id = int(row["task_name"].split(".")[-1])
        sites = row["sites"]
        if isinstance(sites, str):
            parts = sites.strip().split()
            mapping[task_id] = parts[0] if len(parts) == 1 else "cross-site"
        elif isinstance(sites, list):
            mapping[task_id] = sites[0] if len(sites) == 1 else "cross-site"
        else:
            mapping[task_id] = "unknown"
    return mapping


def categorize_workarena_task(task_name: str, level: str) -> str:
    """Categorize a WorkArena task name into a group."""
    name = task_name.replace("workarena.servicenow.", "")
    name = re.sub(r"-l[123]$", "", name)

    if level == "l1":
        if name.startswith("create-"):
            return "create"
        if name.startswith("filter-"):
            return "filter"
        if name.startswith("sort-"):
            return "sort"
        if name.startswith("order-"):
            return "order"
        if "chart" in name:
            return "chart"
        return "navigation"
    else:  # l2
        if name.startswith("dashboard-"):
            return "dashboard"
        if "filter-problems" in name or "mark-duplicates" in name:
            return "filter-problems"
        if name.startswith("filter-") or name.startswith("priority-filter-"):
            return "filter"
        if name.startswith("infeasible-"):
            return "infeasible"
        if name.startswith("navigate-"):
            return "navigate"
        if "expense-management" in name or name.startswith("easy-") or name.startswith("high-"):
            return "expense-mgmt"
        if name.startswith("work-assignment") or name.startswith("workload"):
            return "work-assign"
        if name.startswith("three-") or name.startswith("two-"):
            return "multi-channel"
        if name.startswith("on-") or name.startswith("off-"):
            return "other"
        if name.startswith("get-"):
            return "other"
        return "other"


def load_site_mapping_workarena(level: str) -> dict[str, str]:
    """Build task_name -> category mapping for WorkArena."""
    from browsergym.experiments.benchmark.base import Benchmark
    from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
    from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds
    from browsergym.experiments.benchmark.metadata.utils import task_metadata, task_list_from_metadata

    tm = task_metadata("workarena")
    if level == "l1":
        action_set_key = "workarena"
        max_steps = 15
    else:
        action_set_key = "workarena++"
        max_steps = 50

    bm = Benchmark(
        name=f"workarena_{level}",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS[action_set_key],
        is_multi_tab=(level != "l1"),
        supports_parallel_seeds=False,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=task_list_from_metadata(tm, filter={"level": f"^{level}$"}),
            max_steps=max_steps,
            fixed_seeds=list(range(10)) if level == "l1" else [0],
        ),
        task_metadata=tm,
    )
    split = "full" if level == "l1" else "test"
    if split != "full":
        bm = bm.subset_from_split(split)

    mapping = {}
    for env_args in bm.env_args_list:
        task_name = str(env_args.task_name)
        mapping[task_name] = categorize_workarena_task(task_name, level)
    return mapping


# ── MiniWoB categorization ──

_MINIWOB_WORKFLOW_TASKS = frozenset({
    "book-flight", "book-flight-nodelay", "buy-ticket", "order-food",
})
_MINIWOB_TYPE_TASKS = frozenset({
    "login-user", "login-user-popup", "sign-agreement",
    "multi-layouts", "multi-orderings", "choose-list",
    "highlight-text", "highlight-text-2", "terminal",
})
_MINIWOB_COMPREHEND_TASKS = frozenset({
    "read-table", "read-table-2", "scroll-text", "scroll-text-2",
    "search-engine", "phone-book", "stock-market", "navigate-tree",
    "find-word", "find-greatest", "ascending-numbers",
})
_MINIWOB_CLICK_EXTRAS = frozenset({"unicode-test"})


def categorize_miniwob_task(task_name: str) -> str:
    """Categorize a MiniWoB task by primary web interaction capability."""
    name = task_name.replace("miniwob.", "")
    prefix = name.split("-")[0]

    if name in _MINIWOB_CLICK_EXTRAS:
        return "click"
    if name in _MINIWOB_WORKFLOW_TASKS:
        return "workflow"
    if name in _MINIWOB_TYPE_TASKS:
        return "type"
    if name in _MINIWOB_COMPREHEND_TASKS:
        return "comprehend"

    if prefix == "click":
        return "click"
    if prefix in ("drag", "draw"):
        return "drag"
    if prefix in ("email", "social"):
        return "workflow"
    if prefix == "use" or (prefix == "choose" and "date" in name) or name == "daily-calendar":
        return "widget"
    if prefix in ("enter", "copy", "focus", "form", "text", "resize"):
        return "type"

    return "reason"


def load_site_mapping_miniwob() -> dict[str, str]:
    """Load task_name -> functional category mapping."""
    from browsergym.experiments.benchmark.metadata.utils import task_metadata

    df = task_metadata("miniwob")
    return {row["task_name"]: categorize_miniwob_task(row["task_name"]) for _, row in df.iterrows()}


def load_site_mapping_assistantbench() -> dict[int, str]:
    """Load validation_id -> difficulty mapping."""
    from browsergym.experiments.benchmark.metadata.utils import task_metadata

    df = task_metadata("assistantbench")
    valid = df[df["browsergym_split"] == "valid"]
    return {int(row["task_name"].rsplit(".", 1)[-1]): row["difficulty"] for _, row in valid.iterrows()}


# ═══════════════════════════════════════════════════════════════
# Task key extraction
# ═══════════════════════════════════════════════════════════════

def extract_task_id(task_subdir_name: str) -> int:
    """Extract numeric task ID from a directory name like '..._on_webarena.123_0'."""
    m = re.search(r"\.(\d+)_\d+$", task_subdir_name)
    if m:
        return int(m.group(1))
    m = re.search(r"\.(\d+)$", task_subdir_name)
    if m:
        return int(m.group(1))
    return -1


def extract_task_key(task_subdir_name: str, benchmark: str) -> str | int:
    """Extract task identifier from a task subdir name."""
    if benchmark.startswith("WorkArena") or benchmark == "MiniWoB":
        m = re.search(r"_on_(workarena\.servicenow\..+?)_\d+$", task_subdir_name)
        if m:
            return m.group(1)
        parts = task_subdir_name.split("_on_")
        if len(parts) > 1:
            return re.sub(r"_\d+(_\d+)?$", "", parts[-1])
        return task_subdir_name
    elif benchmark == "AssistantBench":
        m = re.search(r"_on_assistantbench\.validation\.(\d+)_\d+", task_subdir_name)
        if m:
            return int(m.group(1))
        return task_subdir_name
    else:
        return extract_task_id(task_subdir_name)


# ═══════════════════════════════════════════════════════════════
# Reward loading
# ═══════════════════════════════════════════════════════════════

def get_summary_info_list(subdir: Path) -> list[Path]:
    """Recursively find all summary_info.json files, excluding _-prefixed dirs."""
    results = []
    for d in subdir.iterdir():
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        sf = d / "summary_info.json"
        if sf.exists():
            results.append(sf)
    return results


def _count_summary_infos(subdir: Path) -> int:
    count = 0
    def walk(dirpath):
        nonlocal count
        with os.scandir(dirpath) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    if not entry.name.startswith("_"):
                        walk(entry.path)
                elif entry.is_file() and entry.name == "summary_info.json":
                    count += 1
    walk(subdir)
    return count


def _load_cache(cache_path: Path) -> list | None:
    try:
        with open(cache_path, "rb") as f:
            return orjson.loads(f.read())
    except (OSError, orjson.JSONDecodeError):
        return None


def _build_and_save_cache(subdir: Path, cache_path: Path, benchmark: str) -> list:
    summary_infos = get_summary_info_list(subdir)
    entries = []
    for info_path in summary_infos:
        try:
            task_key = extract_task_key(info_path.parent.name, benchmark)
            with open(info_path, "rb") as f:
                summary = orjson.loads(f.read())
            entries.append({
                "task_key": task_key,
                "reward": summary["cum_reward"],
                "has_error": bool(summary.get("err_msg")),
                "cost": summary.get("stats.cum_cost") or 0.0,
                "input_tokens": summary.get("stats.cum_input_tokens") or 0,
                "output_tokens": summary.get("stats.cum_output_tokens") or 0,
            })
        except Exception as e:
            print(f"  Warning: failed to load {info_path}: {e}")
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(orjson.dumps(entries))
    return entries


def load_rewards(subdir: Path, site_mapping: dict, benchmark: str = "") -> dict:
    """Load rewards from summary_info.json files, grouped by site/category.

    Returns dict with keys: 'all', 'by_site', 'count', 'total_tasks', 'errors'.
    """
    rewards = {
        "all": [],
        "by_site": defaultdict(list),
        "count": 0,
        "total_tasks": 0,
        "errors": 0,
        "total_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    task_dirs = [d for d in subdir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]
    rewards["total_tasks"] = len(task_dirs)

    cache_path = subdir / ".cache" / "rewards.json"
    entries = _load_cache(cache_path)
    actual_count = _count_summary_infos(subdir)
    if entries is None or len(entries) != actual_count:
        entries = _build_and_save_cache(subdir, cache_path, benchmark)

    for entry in entries:
        reward = entry["reward"]
        rewards["all"].append(reward)
        rewards["count"] += 1
        if entry["has_error"]:
            rewards["errors"] += 1
        rewards["total_cost"] += entry.get("cost") or 0.0
        rewards["total_input_tokens"] += entry.get("input_tokens") or 0
        rewards["total_output_tokens"] += entry.get("output_tokens") or 0
        site = site_mapping.get(entry["task_key"], "unknown")
        rewards["by_site"][site].append(reward)

    return rewards


def avg(lst: list) -> float:
    if not lst:
        return 0.0
    return sum(lst) / len(lst)
