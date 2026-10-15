# WebArena and WorkArena++ L2 instruction lengths

The **BrowserGym test split** has an average of **125.00 words** per WorkArena++
L2 instruction (185 tasks), versus **14.95 words** per WebArena instruction
(381 tasks): **8.36 times longer**. These are task instructions, not model input
tokens or trajectory lengths.

## Results

| Benchmark | Partition | Instructions / expected tasks | Total words | Mean words |
|---|---|---:|---:|---:|
| WebArena | test | 381 / 381 | 5,695 | 14.947507 |
| WebArena | train | 431 / 431 | 6,326 | 14.677494 |
| WebArena | full | 812 / 812 | 12,021 | 14.804187 |
| WorkArena++ L2 | test | 185 / 185 | 23,125 | 125.000000 |
| WorkArena++ L2 | train, observed | 155 / 156 | 20,930 | 135.032258 |
| WorkArena++ L2 | full, observed | 340 / 341 | 44,055 | 129.573529 |

The L2 full-set mean is an **available-instruction mean**, not a complete
341-task benchmark mean. One train episode has an empty initial goal:
`workarena.servicenow.filter-requested-items-and-order-loaner-laptop-l2`.
It remains an explicit missing record and does not contribute a zero.
Both test-split means have complete coverage.

## Counting and selection

- A word is a Unicode-whitespace-delimited unit: `len(instruction.split())`.
  This includes numbers, standalone list markers, and punctuation-only units;
  hyphenated compounds without whitespace count as one. This is an explicit
  whitespace word count, not a linguistic tokenizer or an LLM tokenizer.
- Count each task once at seed 0, regardless of success, number of steps, or
  instruction duplication across different tasks. Do not pool model reruns.
- WebArena instructions come from the official `intent` field, which the
  BrowserGym adapter passes as the goal. All 812 intents were checked against
  the installed version used by the local evaluations: they match exactly.
- L2 instructions come from `step_0.obs.goal` in the original A3-Qwen3.5-9B
  evaluation studies. The script supports a text-only `goal_object` fallback.
  It excludes observations, accessibility trees, screenshots, system prompts,
  assistant messages, and later-step repetitions.
- The full L2 goal includes its title, introductory sentence, numbered substeps,
  and task parameters. Retaining the substeps matters: the official L2
  implementation supplies them to the agent.
- Split membership comes from pinned BrowserGym metadata, not directory names
  or an inference from success rates. Duplicate task/seed records, unexpected
  task IDs, nontext goals, and inconsistent splits fail rather than being
  silently discarded. Dot/underscore-prefixed caches and archived retries are
  excluded.

Instruction length is descriptive; it does not by itself establish task
difficulty or explain the performance difference.

## Public data coverage and reproducibility limits

The [public A3 trajectory release](https://huggingface.co/datasets/xhluca/a3-qwen-3.5-9b-trajectories/tree/212dc1c418d53b670fd796349817dc8e26fffbdc)
at revision `212dc1c418d53b670fd796349817dc8e26fffbdc` contains **153 L2
episode records, all from the train split**, with **152 nonempty instructions**.
Their mean is **135.848684 words**. All 152 available instruction hashes match
the corresponding local A3 train episodes; none differ.

That release contains **zero of the 185 L2 test instructions**, so downloading
it alone cannot reproduce the test-split extraction. The committed
[count snapshot](../analysis/benchmark_instruction_lengths.json) permits
independent recomputation of every reported mean, total, denominator, and missing
task list. It includes per-task counts, exact-text SHA-256 hashes, source-file
hashes, split metadata, and source paths/URLs, but **no private instruction text,
observations, credentials, or raw traces**. Full L2 extraction requires the
original local study artifacts identified below. Arithmetic reproducibility
from the snapshot is distinct from public reproducibility of raw extraction.

## Reproduce

Offline recomputation and consistency checks use only the Python standard
library:

```sh
python scripts/analyze_benchmark_instruction_lengths.py \
  --snapshot analysis/benchmark_instruction_lengths.json
python -m unittest discover -s tests -p 'test_benchmark_instruction_lengths.py'
```

Re-download the pinned public sources and recompute the **public-only** result
(no authentication, browser instance, or model needed; approximately 72 MB of
initial-step data, not entire trajectories):

```sh
python scripts/analyze_benchmark_instruction_lengths.py \
  --cache-dir /tmp/a3-instruction-length-cache \
  --output /tmp/a3-public-instruction-lengths.json
```

Reproduce the full analysis from **trusted local A3 logs**, with NumPy installed,
and audit the public release:

```sh
python scripts/analyze_benchmark_instruction_lengths.py \
  --results-root agentlab_results \
  --trust-local-pickles \
  --audit-public \
  --cache-dir /tmp/a3-instruction-length-cache \
  --output /tmp/a3-instruction-lengths.json
```

**Never enable the pickle trust flag for untrusted downloads.** The compatibility
reader avoids importing the old AgentLab runtime, but is not a security sandbox.
The public-only and offline-snapshot modes never unpickle anything.

Local studies (relative to `--results-root`; these are the same A3 studies used
by the evaluation-significance script):

- Test: `workarena_l2_test/2026-03-14_12-38-08_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-workarena-l2-test-test`.
- Train: `workarena_l2_test/2026-04-06_18-39-04_genericagent-checkpoints-qwen-qwen3-5-9b-web-pro-low-8903051-checkpoint-latest-on-workarena-l2-train-train`.

Public sources:

- [Official WebArena task intents, pinned revision](https://github.com/web-arena-x/webarena/blob/bb6e4c623e73b5b5ce3caeef82e00d3853de8189/config_files/test.raw.json).
- [BrowserGym WebArena split metadata, pinned revision](https://github.com/ServiceNow/BrowserGym/blob/d7810f9da730514a64789c7920a3a63a8e83339a/browsergym/experiments/src/browsergym/experiments/benchmark/metadata/webarena.csv).
- [BrowserGym WorkArena level/split metadata, pinned revision](https://github.com/ServiceNow/BrowserGym/blob/d7810f9da730514a64789c7920a3a63a8e83339a/browsergym/experiments/src/browsergym/experiments/benchmark/metadata/workarena.csv).

The script verifies SHA-256 digests for all three fixed source files, including
cached copies. Public trajectory JSONs are fetched at a pinned dataset commit
and their file hashes are recorded in the snapshot. The analysis does not
modify the manuscript, run experiments, or publish private evaluation traces.
