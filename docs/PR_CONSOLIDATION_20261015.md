# Public repository consolidation — 2026-10-15

## Preserved and integrated versions

- `legacy` preserves the existing local analysis branch at `b5f04db5e50a5c20de85118daef35b7045bf48c2` (PR #11). It is not a copy of main: pre-integration main is `79b90b45a8ef1e93a6edeb65a8d565c9cee17c77`, also retained in history.
- `codex/integrate-public-prs-20261015` starts from that main and merges #9 and #11 with their original commit histories. No force pushes or branch deletion.

## What is merged

| PR | Contents | Compatibility |
| --- | --- | --- |
| #9 | 50-step exploration default; optional L1–L5 schema, prompts and validation; trajectory conversion compatibility; optional non-greedy collection | GenericAgent remains the public explorer. Unlevelled legacy records stay unlevelled; use `--max-steps 20` for the previous exploration budget. Collection temperature behavior changes only with `--no-reproducibility-mode`. |
| #11 | Reproducible instruction-length extraction, count-only snapshots, methodology and tests | Retains the documented missing full-set instruction and absence of public L2 test goals. No private raw trajectory data added. |

These PRs have no overlapping files or duplicate commits requiring removal. They were merged into one integration branch before testing.

## Integration fixes

- Eliminate three copies of the incorrect `cli/configs/model_configs.json` lookup. Eval, explore and synth now call the same default loader as shorthand resolution.
- Registry lookup is independent of working directory in a source checkout. Set `A3_MODEL_CONFIGS=/absolute/path/to/model_configs.json` for a custom registry or a wheel install without the root `configs/` directory. An explicit loader argument takes precedence. Missing registries give an actionable error.
- Replace the unregistered CLI default `qwen3-vl-32b-thinking` with the shipped `qwen3.5-9b`. Explicit `-m` selections are unaffected; custom registries should select their model explicitly.
- Clarify that `TRAINING_STRATEGYAGENT_ARM.md` is a historical private-checkout recipe, not runnable public commands. No private StrategyAgent implementation was copied into this public repository.

## Validation

`python -m unittest discover -s tests`: **60 tests passed**, comprising 52 existing/PR tests, four registry integration tests and four trajectory compatibility tests.

The latter execute the actual property implementations extracted from the source: both model-argument layouts, legacy Discussion conversion, exact assistant-target reconstruction, no duplicate assistant turn, no prompt mutation and no fabricated missing completion. They do not import private agents or publish private data.

CLI `--help` smoke checks for explore, synth and eval were run using an existing Python 3.12 research environment. Syntax checks, strict JSON duplicate-key checks and credential-pattern scanning of changed files passed. This is not a fresh-install or full dependency-matrix test.

No paid model calls, live WebArena collection, training or long-run quality evaluation was performed. The 50-step difficulty scheme is an implemented option, not newly validated evidence of trajectory quality. No manuscript changes or result recalculations are included.
