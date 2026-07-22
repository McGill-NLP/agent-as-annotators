"""Compute coverage and instruction-diversity statistics for A3-Synth.

The input is the flattened SFT file released in the A3-Synth dataset. Each
JSONL row contains one system/user/assistant training example. The script
reports:

1. Unique page paths and query-bearing page states, after normalizing away
   WebArena instance indices.
2. Distinct-1 through Distinct-n for the unique task instructions.
3. The most frequently visited functional pages.

Distinct-n follows Li et al. (2016): unique n-grams divided by total n-grams.
https://aclanthology.org/N16-1014/
"""

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_PATH = Path("A3-Synth/training/train.jsonl")

GOAL_RE = re.compile(r"##\s*Goal:\s*(.*?)\s*#\s*Observation", re.DOTALL)
URL_RE = re.compile(r"\(active tab\):.*?URL:\s*(\S+)", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9']+")


def user_text(row: list[dict]) -> str:
    """Return the text content of a training example's user message."""
    content = row[1]["content"]
    if isinstance(content, str):
        return content
    return " ".join(item.get("text", "") for item in content if isinstance(item, dict))


def normalize_host(host: str) -> str:
    """Collapse WebArena instance indices such as ``-xl-2``."""
    return re.sub(r"-xl-\d+", "", host)


def is_landing_page(path: str) -> bool:
    """Identify entry pages excluded from the functional-page ranking."""
    if path in ("", "/"):
        return True
    if "Landing" in path:
        return True
    return path.rstrip("/") == "/admin/admin/dashboard"


def site_for_host(host: str) -> str:
    if "shopping-admin" in host:
        return "Shopping Admin"
    if "shopping" in host:
        return "Shopping"
    if "reddit" in host or "forum" in host:
        return "Reddit"
    if "gitlab" in host:
        return "GitLab"
    if "wikipedia" in host or "wiki" in host:
        return "Wikipedia"
    if "map" in host or "openstreetmap" in host:
        return "Map"
    return f"Other ({host})"


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return list(zip(*(tokens[offset:] for offset in range(n))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the A3-Synth dataset statistics reported in the paper."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help="flattened A3-Synth training JSONL",
    )
    parser.add_argument(
        "--max-n", type=int, default=4, help="largest n used for Distinct-n"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="number of frequent pages to print"
    )
    args = parser.parse_args()

    goals: set[str] = set()
    steps = 0
    full_urls: set[str] = set()
    path_urls: set[str] = set()
    per_site_full: dict[str, set[str]] = collections.defaultdict(set)
    per_site_path: dict[str, set[str]] = collections.defaultdict(set)
    path_frequency: collections.Counter[str] = collections.Counter()
    functional_frequency: collections.Counter[str] = collections.Counter()

    with args.path.open() as input_file:
        for line in input_file:
            row = json.loads(line)
            text = user_text(row)
            steps += 1

            goal_match = GOAL_RE.search(text)
            if goal_match:
                goals.add(goal_match.group(1).strip())

            url_match = URL_RE.search(text)
            if not url_match:
                continue

            raw_url = url_match.group(1).strip().strip('"').rstrip("\\")
            parsed = urlparse(raw_url)
            host = normalize_host(parsed.netloc)
            if not host:
                continue

            site = site_for_host(host)
            path_url = f"{parsed.scheme}://{host}{parsed.path}"
            full_url = path_url + (f"?{parsed.query}" if parsed.query else "")
            full_urls.add(full_url)
            path_urls.add(path_url)
            per_site_full[site].add(full_url)
            per_site_path[site].add(path_url)
            path_frequency[path_url] += 1
            if not is_landing_page(parsed.path):
                functional_frequency[path_url] += 1

    if not goals:
        raise ValueError(f"No task goals found in {args.path}")

    print(f"Source: {args.path}")
    print(f"Steps: {steps:,}")
    print(f"Distinct instructions (trajectories): {len(goals):,}")
    print(f"Average steps per trajectory: {steps / len(goals):.2f}\n")

    print("=== 1. Website-state coverage ===")
    print(f"Unique page paths (no query): {len(path_urls):,}")
    print(f"Unique URLs (path + query):   {len(full_urls):,}\n")
    print(f"{'Site':<16}{'Unique URLs':>12}{'Unique paths':>14}")
    for site in sorted(per_site_full, key=lambda name: -len(per_site_full[name])):
        print(
            f"{site:<16}{len(per_site_full[site]):>12,}"
            f"{len(per_site_path[site]):>14,}"
        )

    print(f"\nTop {args.top} pages by step count:")
    for page, count in path_frequency.most_common(args.top):
        print(f"  {count:>5} ({100 * count / steps:4.1f}%)  {page}")

    print(f"\nTop {args.top} functional pages (entry pages excluded):")
    for page, count in functional_frequency.most_common(args.top):
        print(f"  {count:>5} ({100 * count / steps:4.1f}%)  {page}")

    print("\n=== 2. Instruction n-gram diversity ===")
    sorted_goals = sorted(goals)
    lengths = [len(TOKEN_RE.findall(goal.lower())) for goal in sorted_goals]
    print(
        "Instruction length (words): "
        f"mean={statistics.mean(lengths):.1f} "
        f"median={statistics.median(lengths):g} "
        f"min={min(lengths)} max={max(lengths)}"
    )
    print(f"{'Metric':<20}{'Unique':>10}{'Total':>10}{'Distinct-n':>12}")
    for n in range(1, args.max_n + 1):
        unique_ngrams: set[tuple[str, ...]] = set()
        total = 0
        for goal in sorted_goals:
            goal_ngrams = ngrams(TOKEN_RE.findall(goal.lower()), n)
            total += len(goal_ngrams)
            unique_ngrams.update(goal_ngrams)
        label = "Distinct-1 (vocab)" if n == 1 else f"Distinct-{n}"
        print(
            f"{label:<20}{len(unique_ngrams):>10,}{total:>10,}"
            f"{len(unique_ngrams) / total:>12.4f}"
        )


if __name__ == "__main__":
    main()
