# Training a StrategyAgent-collected arm

*(Written from the A3S5 run in the private llm-annotators repo; script names there are given in parentheses. The traps are properties of this pipeline, not of that one run.)*

End-to-end, with the traps that actually bite. Every command here has been run.

## TL;DR

```bash
# 1. collect (already done for A3S5)
bash scripts/run_a3s5_collection.sh xl-0 xl-1 xl-2 xl-3 xl-4 xl-5

# 2. drive the collection to CLEAN, not merely "attempted"
bash scripts/a3s5_recover_until_clean.sh

# 3. build the SFT dataset (manifest-gated)
bash scripts/build_a3s5_sft.sh
#    -> trajectories/rft_data_a3s5/a3s5-strategy-flash37-stratified/train.jsonl

# 4. train
#    edit configs/train/a3s5-qwen3.5-9b-tamia.json -> data_path = the train.jsonl above
sbatch slurm/job-train-a3s5-qwen3.5-9b-tamia.sh
```

## The five traps, in the order you will hit them

### 1. The in-repo runner uses the WRONG AGENT by default

`run_websynth.py` builds `GenericAgentArgs`. **Every A3S arm was collected with StrategyAgent
v2.2.** Pass `--agent strategyagent`, or you silently collect with a different scaffold from
the arms you will compare against. Nothing errors; only the study directory name differs.

### 2. `reproducibility_mode` forces temperature 0.0

Also default-on in `run_websynth.py`. Right for an eval, **wrong for a collection** — greedy
decoding collapses the trajectory diversity the collection exists to produce. A3S2 collected at
temp 1.0 / top_p 1.0. Pass `--no-reproducibility-mode`.

### 3. The trajectory pipeline assumed GenericAgent (fixed, but know why)

`llm_annotators/utils/trajectories.py` reached for GenericAgent-only attributes and raised on
every StrategyAgent trajectory. Three separate ones, each aborting the whole conversion:

| access | GenericAgent | StrategyAgent | fix |
|---|---|---|---|
| `agent_args.chat_model_args.model_name` | nested | `model_name` directly on args | fall back to the args object |
| `agent_args.flags.asdict()` | flags dataclass | no `flags` field | return `{}` |
| `agent_info.chat_messages.to_openai()` | `Discussion` object | plain list, already OpenAI format | use as-is when no `to_openai` |

### 4. Staging must use HARDLINKS, not symlinks

`list_experiments()` selects directories with `rglob("**/summary_info.json")`, and pathlib's
**rglob does not follow symlinks**. A symlinked staging tree stages fine and then converts
**zero** trajectories. Use `cp -al` — real directories, hardlinked files, no data duplicated.

### 5. `generate_rft_data` filters on the JUDGE, so it cannot remove judge errors

`scripts/generate_rft_data.py:110` keeps a trajectory iff `cum_reward > 0`. That removes
failures, **not mislabels**. Measured on A3S5: 46 episodes whose every step was a 502 Bad
Gateway page were scored 1.0 and would pass this filter — training the model to answer from
memory when a site is down. **The manifest, not the judge, is the quality gate.**

## Why the manifest exists

`scripts/build_a3s5_manifest.py` decides what enters training and reconciles explicitly
(prints `DOES NOT RECONCILE` on any mismatch). It excludes what no glob can distinguish:

- the **smoke study** (`*a3s5*` matches it; it ran at temperature 0.0) — the leading dash in
  `*-a3s5-on-websynth*` is load-bearing;
- **dead-site episodes** (gateway-error pages), found by content, not status;
- **stale tasks** from explorations later re-run as too short;
- **quota-walled episodes**, which record `err_msg=None` and read as short-but-clean.

## Collection health: three failures that all look like SPEED

A busted episode finishes instantly, so each of these makes a run appear **faster**, and none
appear in an `err_msg` tally:

| failure | signature | detection |
|---|---|---|
| shared-quota wall | `cum_busted_retry >= 1`, 0-1 steps | strict grep: `RESOURCE_EXHAUSTED\|Error code: 429` — **never bare `429`**, it matches task ids and file paths |
| dead site (roving Kiwix) | every step an error page, judge may score 1.0 | scan step AXTrees for `bad gateway`; probe the **Kiwix deep link**, not the site root (the root answers 200 while the ZIM is down) |
| parse bust | `terminated=False, truncated=True, err_msg=None` | `n_steps < max_steps AND stats.cum_busted_retry >= 1` |

Unexplained acceleration is a failure signal, not a success one.

## Verifying before you spend a GPU-day

```bash
uv run python scripts/gate_websynth_collection.py --self-test            # all 8 detectors fire AND stay silent
uv run python scripts/gate_websynth_collection.py --glob '<root>/*-a3s5-on-websynth*'
uv run python scripts/mark_dead_site_episodes.py                         # dry run
uv run python scripts/build_a3s5_manifest.py                             # must print ACCOUNTED n/n
wc -l trajectories/rft_data_a3s5/*/train.jsonl                           # A3S4 ref: 7,725 from 861 episodes
```

Thinking lives **inline in `content` as `<thought>`**, not in `reasoning_content` (empty on
120/120 sampled steps). Verify thinking survives by grepping the captured content, not by
inspecting the field you expect to be populated.
