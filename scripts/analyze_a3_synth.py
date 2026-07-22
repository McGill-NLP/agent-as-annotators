"""Compute reproducible coverage and instruction statistics for A3-Synth.

The input is the released flattened SFT JSONL file. Each row represents one
training step and contains a user message with a ``## Goal`` block and an
active-tab URL. The implementation intentionally uses only Python's standard
library so the analysis can run without installing the training stack.

Methodology and download instructions are documented in
``docs/dataset_statistics.md``.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

GOAL_RE = re.compile(r"##\s*Goal:\s*(.*?)\s*#\s*Observation", re.DOTALL)
URL_RE = re.compile(r"\(active tab\):.*?URL:\s*(\S+)", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9']+")
INSTANCE_RE = re.compile(r"-xl-\d+(?=\.|:|$)")


class DatasetFormatError(ValueError):
    """Raised when a JSONL row does not match the released SFT format."""


@dataclass(frozen=True)
class SiteCoverage:
    site: str
    unique_urls: int
    unique_paths: int


@dataclass(frozen=True)
class PageFrequency:
    url: str
    site: str
    steps: int
    share: float


@dataclass(frozen=True)
class NgramDiversity:
    n: int
    unique: int
    total: int
    distinct_n: float


@dataclass(frozen=True)
class InstructionLengths:
    mean: float
    median: float
    minimum: int
    maximum: int


@dataclass(frozen=True)
class DatasetStatistics:
    source: str
    steps: int
    steps_with_web_url: int
    steps_without_web_url: int
    distinct_instructions: int
    average_steps_per_instruction: float
    unique_paths: int
    unique_urls: int
    site_coverage: tuple[SiteCoverage, ...]
    top_pages: tuple[PageFrequency, ...]
    top_functional_pages: tuple[PageFrequency, ...]
    instruction_lengths: InstructionLengths
    ngram_diversity: tuple[NgramDiversity, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with a stable schema."""
        return asdict(self)


def positive_integer(value: str) -> int:
    """Argparse converter requiring an integer greater than zero."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def user_text(row: Any, line_number: int) -> str:
    """Extract text from the user message of one flattened SFT row."""
    if not isinstance(row, list) or len(row) < 2:
        raise DatasetFormatError(
            f"line {line_number}: expected a message list with at least two items"
        )
    user_message = row[1]
    if not isinstance(user_message, dict) or "content" not in user_message:
        raise DatasetFormatError(
            f"line {line_number}: second message has no content field"
        )

    content = user_message["content"]
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise DatasetFormatError(
            f"line {line_number}: user content must be a string or a list"
        )

    parts = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    if not parts:
        raise DatasetFormatError(
            f"line {line_number}: user content contains no text parts"
        )
    return " ".join(parts)


def normalize_host(netloc: str) -> str:
    """Lowercase a host and remove the WebArena ``-xl-N`` instance index."""
    return INSTANCE_RE.sub("", netloc.lower())


def normalize_page_url(raw_url: str, line_number: int) -> tuple[str, str, str] | None:
    """Return ``(site, origin+path, origin+path+query)`` for an observed URL.

    Fragments are discarded. Queries are retained only in the full URL. The
    WebArena instance suffix is removed from the network location in both
    representations.
    """
    cleaned = raw_url.strip().strip('"').rstrip("\\")
    parsed = urlsplit(cleaned)
    if parsed.scheme.lower() == "about" and parsed.path == "blank":
        return None
    if not parsed.scheme or not parsed.netloc:
        raise DatasetFormatError(
            f"line {line_number}: active-tab URL is not absolute: {cleaned!r}"
        )
    netloc = normalize_host(parsed.netloc)
    path_url = urlunsplit((parsed.scheme.lower(), netloc, parsed.path, "", ""))
    full_url = urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path, parsed.query, "")
    )
    return site_for_host(netloc), path_url, full_url


def is_landing_page(path: str) -> bool:
    """Identify entry pages excluded from the functional-page ranking."""
    if path in ("", "/"):
        return True
    if "Landing" in path:
        return True
    return path.rstrip("/") == "/admin/admin/dashboard"


def site_for_host(host: str) -> str:
    """Map released WebArena hosts to the six paper site labels."""
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


def ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    """Yield within-instruction n-grams without crossing goal boundaries."""
    for start in range(len(tokens) - n + 1):
        yield tuple(tokens[start : start + n])


def ranked_pages(
    frequencies: collections.Counter[str], steps: int, limit: int
) -> tuple[PageFrequency, ...]:
    """Rank pages deterministically by descending count then normalized URL."""
    pages = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return tuple(
        PageFrequency(
            url=url,
            site=site_for_host(urlsplit(url).netloc),
            steps=count,
            share=count / steps,
        )
        for url, count in pages
    )


def analyze_dataset(path: Path, max_n: int = 4, top: int = 10) -> DatasetStatistics:
    """Analyze one released flattened training JSONL file."""
    if max_n < 1:
        raise ValueError("max_n must be at least 1")
    if top < 1:
        raise ValueError("top must be at least 1")

    goals: set[str] = set()
    full_urls: set[str] = set()
    path_urls: set[str] = set()
    per_site_full: dict[str, set[str]] = collections.defaultdict(set)
    per_site_path: dict[str, set[str]] = collections.defaultdict(set)
    page_frequency: collections.Counter[str] = collections.Counter()
    functional_frequency: collections.Counter[str] = collections.Counter()
    steps = 0
    steps_with_web_url = 0

    try:
        input_file = path.open(encoding="utf-8")
    except OSError as error:
        raise DatasetFormatError(f"cannot open {path}: {error.strerror}") from error

    with input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise DatasetFormatError(f"line {line_number}: blank lines are invalid")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetFormatError(
                    f"line {line_number}: invalid JSON ({error.msg})"
                ) from error

            text = user_text(row, line_number)
            goal_match = GOAL_RE.search(text)
            if not goal_match or not goal_match.group(1).strip():
                raise DatasetFormatError(f"line {line_number}: task goal not found")
            url_match = URL_RE.search(text)
            if not url_match:
                raise DatasetFormatError(
                    f"line {line_number}: active-tab URL not found"
                )

            goal = goal_match.group(1).strip()
            steps += 1
            goals.add(goal)
            normalized_url = normalize_page_url(url_match.group(1), line_number)
            if normalized_url is None:
                continue
            site, path_url, full_url = normalized_url
            steps_with_web_url += 1
            full_urls.add(full_url)
            path_urls.add(path_url)
            per_site_full[site].add(full_url)
            per_site_path[site].add(path_url)
            page_frequency[path_url] += 1
            if not is_landing_page(urlsplit(path_url).path):
                functional_frequency[path_url] += 1

    if steps == 0:
        raise DatasetFormatError(f"{path}: input contains no rows")

    sorted_goals = sorted(goals)
    tokenized_goals = [TOKEN_RE.findall(goal.lower()) for goal in sorted_goals]
    lengths = [len(tokens) for tokens in tokenized_goals]
    ngram_rows = []
    for n in range(1, max_n + 1):
        unique_ngrams: set[tuple[str, ...]] = set()
        total = 0
        for tokens in tokenized_goals:
            goal_ngrams = list(ngrams(tokens, n))
            total += len(goal_ngrams)
            unique_ngrams.update(goal_ngrams)
        ngram_rows.append(
            NgramDiversity(
                n=n,
                unique=len(unique_ngrams),
                total=total,
                distinct_n=len(unique_ngrams) / total if total else 0.0,
            )
        )

    coverage = tuple(
        SiteCoverage(
            site=site,
            unique_urls=len(per_site_full[site]),
            unique_paths=len(per_site_path[site]),
        )
        for site in sorted(
            per_site_full,
            key=lambda name: (-len(per_site_full[name]), name),
        )
    )

    return DatasetStatistics(
        source=str(path),
        steps=steps,
        steps_with_web_url=steps_with_web_url,
        steps_without_web_url=steps - steps_with_web_url,
        distinct_instructions=len(goals),
        average_steps_per_instruction=steps / len(goals),
        unique_paths=len(path_urls),
        unique_urls=len(full_urls),
        site_coverage=coverage,
        top_pages=ranked_pages(page_frequency, steps, top),
        top_functional_pages=ranked_pages(functional_frequency, steps, top),
        instruction_lengths=InstructionLengths(
            mean=statistics.mean(lengths),
            median=statistics.median(lengths),
            minimum=min(lengths),
            maximum=max(lengths),
        ),
        ngram_diversity=tuple(ngram_rows),
    )


def render_text(stats: DatasetStatistics) -> str:
    """Render the human-readable report used to verify paper values."""
    lines = [
        f"Source: {stats.source}",
        f"Steps: {stats.steps:,}",
        f"Steps with webpage URL: {stats.steps_with_web_url:,}",
        f"Steps without webpage URL: {stats.steps_without_web_url:,}",
        f"Distinct instructions: {stats.distinct_instructions:,}",
        f"Average steps per instruction: {stats.average_steps_per_instruction:.2f}",
        "",
        "=== 1. Website-state coverage ===",
        f"Unique page paths (no query): {stats.unique_paths:,}",
        f"Unique URLs (path + query):   {stats.unique_urls:,}",
        "",
        f"{'Site':<16}{'Unique URLs':>12}{'Unique paths':>14}",
    ]
    for coverage_row in stats.site_coverage:
        lines.append(
            f"{coverage_row.site:<16}{coverage_row.unique_urls:>12,}"
            f"{coverage_row.unique_paths:>14,}"
        )

    lines.extend(["", f"Top {len(stats.top_pages)} pages by step count:"])
    for page in stats.top_pages:
        lines.append(f"  {page.steps:>5} ({100 * page.share:4.1f}%)  {page.url}")

    lines.extend(
        [
            "",
            f"Top {len(stats.top_functional_pages)} functional pages "
            "(entry pages excluded):",
        ]
    )
    for page in stats.top_functional_pages:
        lines.append(f"  {page.steps:>5} ({100 * page.share:4.1f}%)  {page.url}")

    lengths = stats.instruction_lengths
    lines.extend(
        [
            "",
            "=== 2. Instruction n-gram diversity ===",
            "Instruction length (words): "
            f"mean={lengths.mean:.1f} median={lengths.median:g} "
            f"min={lengths.minimum} max={lengths.maximum}",
            f"{'Metric':<20}{'Unique':>10}{'Total':>10}{'Distinct-n':>12}",
        ]
    )
    for ngram_row in stats.ngram_diversity:
        label = "Distinct-1 (vocab)" if ngram_row.n == 1 else f"Distinct-{ngram_row.n}"
        lines.append(
            f"{label:<20}{ngram_row.unique:>10,}{ngram_row.total:>10,}"
            f"{ngram_row.distinct_n:>12.4f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute reproducible A3-Synth dataset statistics."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="released flattened SFT JSONL (for example, training/train.jsonl)",
    )
    parser.add_argument(
        "--max-n",
        type=positive_integer,
        default=4,
        help="largest n used for Distinct-n (default: 4)",
    )
    parser.add_argument(
        "--top",
        type=positive_integer,
        default=10,
        help="number of frequent pages to report (default: 10)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        stats = analyze_dataset(args.input, max_n=args.max_n, top=args.top)
    except (DatasetFormatError, ValueError) as error:
        parser.error(str(error))

    if args.format == "json":
        print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
