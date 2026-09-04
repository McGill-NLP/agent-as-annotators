"""Count task-instruction words, with explicit split coverage and provenance.

No model inference or benchmark environments are run. Public-only extraction
uses pinned, unauthenticated sources. Local extraction requires an explicit
trust flag for pickle files; only lengths and hashes are exported, never goals,
observations, credentials, or model messages.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import pickle
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

BG_REV = "d7810f9da730514a64789c7920a3a63a8e83339a"
WA_REV = "bb6e4c623e73b5b5ce3caeef82e00d3853de8189"
HF_REPO = "xhluca/a3-qwen-3.5-9b-trajectories"
HF_REV = "212dc1c418d53b670fd796349817dc8e26fffbdc"
BG_ROOT = f"https://raw.githubusercontent.com/ServiceNow/BrowserGym/{BG_REV}/browsergym/experiments/src/browsergym/experiments/benchmark/metadata"
SOURCES = {
    "webarena_tasks": (
        f"https://raw.githubusercontent.com/web-arena-x/webarena/{WA_REV}/config_files/test.raw.json",
        "7b50386fd69163dbc05d615d834df4c6ed2c35596e97a1b10d17451c02537652",
    ),
    "webarena_metadata": (
        f"{BG_ROOT}/webarena.csv",
        "7f93d2a01bc9b3704eb26314853b93626a365385e14ca72a87aa4c883d8fd920",
    ),
    "workarena_metadata": (
        f"{BG_ROOT}/workarena.csv",
        "013b99b2712abe931c792083b10ed9e0cb097a64215650d2700d3b0541629a47",
    ),
}
LOCAL_STUDIES = {
    "test": "workarena_l2_test/2026-03-14_12-38-08_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-workarena-l2-test-test",
    "train": "workarena_l2_test/2026-04-06_18-39-04_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-workarena-l2-train-train",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, cache, expected=None):
    """Cache by URL; verify fixed-source digests, including on cache hits."""
    path = cache / sha256(url.encode())
    if path.exists():
        data = path.read_bytes()
    else:
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "a3-instruction-audit/1"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if expected and sha256(data) != expected:
            raise ValueError(f"source checksum mismatch: {url}")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if expected and sha256(data) != expected:
        raise ValueError(f"source checksum mismatch: {url}")
    return data


def load_metadata(data, benchmark):
    rows = csv.DictReader(io.StringIO(data.decode()))
    result = {}
    for row in rows:
        if benchmark == "workarena_l2" and row["level"] != "l2":
            continue
        name, split = row["task_name"], row["browsergym_split"]
        if name in result or split not in ("test", "train"):
            raise ValueError("duplicate task or invalid split in metadata")
        result[name] = split
    return result


def instruction(obs):
    """Use the actual initial goal, not the agent prompt or observation text."""
    if obs is None:
        return None
    if not isinstance(obs, dict):
        raise ValueError("observation is not an object")
    goal = obs.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal
    if goal is not None and not isinstance(goal, str):
        raise ValueError("goal is not text")
    parts = obs.get("goal_object")
    if parts is None:
        return None
    if not isinstance(parts, list) or any(
        not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str)
        for p in parts
    ):
        raise ValueError("goal_object is not entirely textual")
    text = "\n".join(p["text"] for p in parts)
    return text if text.strip() else None


def record(benchmark, task, split, seed, goal, source, digest, reason=None):
    return {
        "benchmark": benchmark, "task_name": task, "split": split, "task_seed": seed,
        "word_count": len(goal.split()) if goal else None,
        "instruction_sha256": sha256(goal.encode()) if goal else None,
        "source": source, "source_sha256": digest,
        "missing_reason": reason if not goal else None,
    }


def summarize(records, metadata):
    """One weight per task/seed. Missing instructions never count as zero."""
    seen = set()
    for row in records:
        benchmark, task, seed = row["benchmark"], row["task_name"], row["task_seed"]
        if benchmark not in metadata or task not in metadata[benchmark]:
            raise ValueError("unknown benchmark/task")
        if type(seed) is not int or seed != 0:
            raise ValueError("this audit is defined for seed 0")
        if row["split"] != metadata[benchmark][task]:
            raise ValueError("split does not match pinned metadata")
        key = benchmark, task, seed
        if key in seen:
            raise ValueError(f"duplicate task/seed: {key}")
        seen.add(key)
        count = row["word_count"]
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError("word counts must be positive integers or null")
        if count is None and not row["missing_reason"]:
            raise ValueError("missing instruction needs an explicit reason")
        if count is not None and not re.fullmatch(r"[0-9a-f]{64}", row["instruction_sha256"] or ""):
            raise ValueError("instruction hash missing")
    summaries = {}
    for benchmark, tasks in metadata.items():
        for split in ("test", "train", "all"):
            expected = {t for t, s in tasks.items() if split == "all" or s == split}
            selected = [r for r in records if r["benchmark"] == benchmark and r["task_name"] in expected]
            available = [r for r in selected if r["word_count"] is not None]
            counts = [r["word_count"] for r in available]
            summaries[f"{benchmark}/{split}"] = {
                "expected_tasks": len(expected), "observed_instructions": len(counts),
                "missing_tasks": sorted(expected - {r["task_name"] for r in available}),
                "complete": len(counts) == len(expected),
                "total_words": sum(counts),
                "mean_words": statistics.mean(counts) if counts else None,
                "median_words": statistics.median(counts) if counts else None,
                "min_words": min(counts) if counts else None,
                "max_words": max(counts) if counts else None,
            }
    return summaries


class Record(dict):
    """Inert holder for historical AgentLab data, without importing its stack."""


class LocalReader(pickle.Unpickler):
    """Compatibility reader ONLY for user-trusted local experiment artifacts."""
    def find_class(self, module, name):
        if module.startswith(("agentlab.", "browsergym.")):
            return Record
        return super().find_class(module, name)


def local_records(root, metadata):
    rows = []
    for split, study in LOCAL_STUDIES.items():
        directory = root / study
        if not directory.is_dir():
            raise ValueError(f"missing local study: {directory}")
        for episode in sorted(directory.iterdir()):
            if not episode.is_dir() or episode.name.startswith((".", "_")):
                continue  # AgentLab's archived retries and caches
            match = re.search(r"_on_(.+)_(\d+)$", episode.name)
            if not match:
                raise ValueError(f"unrecognized episode directory: {episode.name}")
            task, seed = match.group(1), int(match.group(2))
            if metadata.get(task) != split:
                raise ValueError(f"unexpected task in {split} study: {task}")
            path = episode / "step_0.pkl.gz"
            goal, digest, reason = None, None, "missing_step_0"
            if path.exists():
                data = path.read_bytes()
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
                    step = LocalReader(handle).load()
                goal = instruction(getattr(step, "obs", step.get("obs")))
                digest, reason = sha256(data), "empty_initial_goal"
            rows.append(record("workarena_l2", task, split, seed, goal,
                               str(path.relative_to(root)), digest, reason))
    return rows


def published_records(cache, metadata, workers):
    listing_url = f"https://huggingface.co/api/datasets/{HF_REPO}/revision/{HF_REV}"
    listing = json.loads(fetch(listing_url, cache))
    if listing["sha"] != HF_REV:
        raise ValueError("unexpected public dataset revision")
    files = {s["rfilename"] for s in listing["siblings"]}
    directories = sorted(p.rsplit("/", 1)[0] for p in files
                         if p.startswith("workarena_l2/") and p.endswith("/exp_args.json"))

    def extract(directory):
        prefix = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/{directory}"
        args = json.loads(fetch(prefix + "/exp_args.json", cache))["env_args"]
        task, seed = args["task_name"], args["task_seed"]
        if task not in metadata:
            raise ValueError(f"unregistered public L2 task: {task}")
        goal, digest, reason = None, None, "missing_step_0"
        url = prefix + "/step_0.json"
        if directory + "/step_0.json" in files:
            data = fetch(url, cache)
            step = json.loads(data)
            if step.get("step") != 0:
                raise ValueError("not an initial step")
            goal = instruction(step.get("obs"))
            digest, reason = sha256(data), "empty_initial_goal"
        return record("workarena_l2", task, metadata[task], seed, goal, url, digest, reason)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(extract, directories))


def build_report(args):
    inputs = {name: fetch(url, args.cache_dir, digest) for name, (url, digest) in SOURCES.items()}
    metadata = {
        "webarena": load_metadata(inputs["webarena_metadata"], "webarena"),
        "workarena_l2": load_metadata(inputs["workarena_metadata"], "workarena_l2"),
    }
    wa = []
    for task in json.loads(inputs["webarena_tasks"]):
        name = f"webarena.{task['task_id']}"
        goal = instruction({"goal": task["intent"]})
        wa.append(record("webarena", name, metadata["webarena"][name], 0, goal,
                         SOURCES["webarena_tasks"][0], SOURCES["webarena_tasks"][1]))
    if args.results_root:
        work = local_records(args.results_root, metadata["workarena_l2"])
    else:
        work = published_records(args.cache_dir, metadata["workarena_l2"], args.workers)
    rows = sorted(wa + work, key=lambda r: (r["benchmark"], r["task_name"], r["task_seed"]))
    report = {
        "schema_version": 1,
        "metric": "len(initial_instruction.split()); Unicode whitespace; retain numbered substeps, numbers, punctuation-only tokens, and hyphenated compounds as supplied",
        "weighting": "one initial instruction per task at seed 0, irrespective of success or trajectory length",
        "source_mode": "official_webarena_and_local_a3_l2" if args.results_root else "official_webarena_and_published_a3_l2",
        "sources": {k: {"url": u, "sha256": h} for k, (u, h) in SOURCES.items()},
        "metadata": metadata, "records": rows, "summary": summarize(rows, metadata),
    }
    if args.results_root:
        report["local_studies"] = LOCAL_STUDIES
    if args.audit_public:
        public = published_records(args.cache_dir, metadata["workarena_l2"], args.workers)
        indexed = {(r["task_name"], r["task_seed"]): r for r in work}
        matches, mismatches = [], []
        for row in public:
            key = row["task_name"], row["task_seed"]
            if row["instruction_sha256"] and key in indexed:
                target = matches if row["instruction_sha256"] == indexed[key]["instruction_sha256"] else mismatches
                target.append(row["task_name"])
        report["public_audit"] = {
            "repo": HF_REPO, "revision": HF_REV, "records": public,
            "summary": summarize(wa + public, metadata),
            "exact_instruction_hash_matches": matches, "instruction_hash_mismatches": mismatches,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--results-root", type=Path, help="A3 AgentLab root for complete test-split extraction")
    mode.add_argument("--snapshot", type=Path, help="recompute and verify statistics from a saved count snapshot, offline")
    parser.add_argument("--trust-local-pickles", action="store_true", help="ONLY for trusted local logs; requires numpy")
    parser.add_argument("--audit-public", action="store_true", help="also check public-release coverage and instruction hashes")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/benchmark-instruction-lengths"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.results_root and not args.trust_local_pickles:
        parser.error("--results-root requires --trust-local-pickles; never load untrusted pickles")
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    if args.snapshot:
        report = json.loads(args.snapshot.read_text())
        if report["summary"] != summarize(report["records"], report["metadata"]):
            raise ValueError("snapshot statistics do not match its per-task counts")
        if "public_audit" in report:
            wa = [r for r in report["records"] if r["benchmark"] == "webarena"]
            audit = report["public_audit"]
            if audit["summary"] != summarize(wa + audit["records"], report["metadata"]):
                raise ValueError("public-audit statistics do not match counts")
    else:
        report = build_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for key, value in report["summary"].items():
        mean = value["mean_words"]
        rendered = f"{mean:.6f}" if mean is not None else "n/a"
        print(f"{key}: {rendered} words; {value['observed_instructions']}/{value['expected_tasks']} tasks")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, pickle.UnpicklingError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
