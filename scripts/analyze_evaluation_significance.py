"""Reproduce the full-benchmark uncertainty and significance table.

For the base and A3-Qwen3.5-9B AgentLab studies, this script computes:

1. Wilson 95% confidence intervals on episode success rates.
2. Exact two-sided McNemar tests on paired ``(task_name, task_seed)`` outcomes.

WebArena, VisualWebArena, and WorkArena++ L2 combine their official train and
test task partitions. WorkArena L1 and MiniWoB use the full benchmark studies.
"""

import argparse
import math
from pathlib import Path

from agentlab.analyze.inspect_results import load_result_df
from scipy.stats import binomtest, norm

STUDIES = [
    (
        "WebArena (812)",
        [
            "webarena_test/2026-03-06_14-56-26_genericagent-qwen-qwen3-5-9b-on-webarena-test-test",
            "webarena_train/2026-03-25_17-04-34_genericagent-qwen-qwen3-5-9b-on-webarena-train-train",
        ],
        [
            "webarena_test/2026-03-10_15-17-39_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-webarena-test-test",
            "webarena_train/2026-03-25_17-01-26_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-webarena-train-train",
        ],
    ),
    (
        "VisualWebArena (910)",
        [
            "visualwebarena_test/2026-03-19_20-26-37_genericagent-qwen-qwen3-5-9b-on-visualwebarena-test-test",
            "visualwebarena_train/2026-04-02_16-00-13_genericagent-qwen-qwen3-5-9b-on-visualwebarena-train-train",
        ],
        [
            "visualwebarena_test/2026-03-20_23-46-04_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-visualwebarena-test-test",
            "visualwebarena_train/2026-04-02_16-56-32_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-visualwebarena-train-train",
        ],
    ),
    (
        "WorkArena L1 (330)",
        [
            "workarena_l1/2026-03-12_20-04-29_genericagent-qwen-qwen3-5-9b-on-workarena-l1-full"
        ],
        [
            "workarena_l1/2026-03-12_18-27-40_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-workarena-l1-full"
        ],
    ),
    (
        "WorkArena++ L2 (341)",
        [
            "workarena_l2_test/2026-03-14_13-30-54_genericagent-qwen-qwen3-5-9b-on-workarena-l2-test-test",
            "workarena_l2_test/2026-04-06_18-39-04_genericagent-qwen-qwen3-5-9b-on-workarena-l2-train-train",
        ],
        [
            "workarena_l2_test/2026-03-14_12-38-08_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-workarena-l2-test-test",
            "workarena_l2_test/2026-04-06_18-39-04_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-workarena-l2-train-train",
        ],
    ),
    (
        "MiniWoB (625)",
        ["miniwob/2026-03-16_16-50-09_genericagent-qwen-qwen3-5-9b-on-miniwob"],
        [
            "miniwob/2026-03-16_16-50-10_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-miniwob"
        ],
    ),
]


def wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    """Return a Wilson 95% confidence interval for a Bernoulli proportion."""
    if episodes == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(0.975)
    proportion = successes / episodes
    denominator = 1 + z * z / episodes
    center = (proportion + z * z / (2 * episodes)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / episodes + z * z / (4 * episodes * episodes)
        )
        / denominator
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def load_episodes(
    results_root: Path, relative_directories: list[str]
) -> list[tuple[tuple[str, object], int]]:
    """Load every episode row as ``((task_name, task_seed), success)``."""
    episodes = []
    for relative_directory in relative_directories:
        directory = results_root / relative_directory
        dataframe = load_result_df(str(directory), progress_fn=None).reset_index()
        for _, row in dataframe.iterrows():
            key = (row["env.task_name"], row.get("env.task_seed"))
            episodes.append((key, int(row["cum_reward"] > 0)))
    return episodes


def paired_outcomes(
    episodes: list[tuple[tuple[str, object], int]],
) -> dict[tuple[str, object], int]:
    """Reduce duplicate task/seed runs to one order-independent outcome."""
    outcomes: dict[tuple[str, object], int] = {}
    for key, success in episodes:
        outcomes[key] = max(outcomes.get(key, 0), success)
    return outcomes


def exact_mcnemar(base_only: int, a3_only: int) -> float:
    discordant = base_only + a3_only
    if discordant == 0:
        return float("nan")
    return binomtest(
        min(base_only, a3_only),
        discordant,
        0.5,
        alternative="two-sided",
    ).pvalue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the A3 full-benchmark significance table."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("agentlab_results"),
        help="directory containing the published AgentLab study directories",
    )
    args = parser.parse_args()

    print("# Full-benchmark Wilson 95% CIs and paired McNemar tests")
    print()
    print(
        "| Benchmark | n_base / n_A3 | Base SR (95% CI) | A3 SR (95% CI) "
        "| Diff | Exact p | Paired n |"
    )
    print("|---|:---:|:---:|:---:|---:|:---:|:---:|")

    for label, base_directories, a3_directories in STUDIES:
        base_episodes = load_episodes(args.results_root, base_directories)
        a3_episodes = load_episodes(args.results_root, a3_directories)
        base_n, a3_n = len(base_episodes), len(a3_episodes)
        base_successes = sum(success for _, success in base_episodes)
        a3_successes = sum(success for _, success in a3_episodes)
        base_rate = base_successes / base_n
        a3_rate = a3_successes / a3_n
        base_low, base_high = wilson_interval(base_successes, base_n)
        a3_low, a3_high = wilson_interval(a3_successes, a3_n)

        base_pairs = paired_outcomes(base_episodes)
        a3_pairs = paired_outcomes(a3_episodes)
        shared = sorted(set(base_pairs) & set(a3_pairs))
        base_only = sum(base_pairs[key] == 1 and a3_pairs[key] == 0 for key in shared)
        a3_only = sum(base_pairs[key] == 0 and a3_pairs[key] == 1 for key in shared)
        p_value = exact_mcnemar(base_only, a3_only)

        n_label = f"{base_n} / {a3_n}" if base_n != a3_n else str(base_n)
        p_label = f"{p_value:.3g}" if not math.isnan(p_value) else "n/a"
        print(
            f"| {label} | {n_label} | "
            f"{100 * base_rate:.1f} [{100 * base_low:.1f}, {100 * base_high:.1f}] | "
            f"{100 * a3_rate:.1f} [{100 * a3_low:.1f}, {100 * a3_high:.1f}] | "
            f"{100 * (a3_rate - base_rate):+.1f} | {p_label} | {len(shared)} |"
        )


if __name__ == "__main__":
    main()
