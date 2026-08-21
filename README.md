<div align="center">

# Agent-as-Annotators (A3)

| [**💾 Code**](https://github.com/McGill-NLP/agent-as-annotators) | [**📄 Paper**](https://arxiv.org/abs/2604.07776) | [**🌐 Website**](https://agent-as-annotators.github.io) |
| :--: | :--: | :--: |
| [**🤗 Dataset**](https://huggingface.co/datasets/McGill-NLP/A3-Synth) | [**🤖 Models**](https://huggingface.co/collections/McGill-NLP/a3-agent-as-annotators-69d854ab5b1993b10efc3fba) | [**📦 PyPI**](https://pypi.org/project/agent-as-annotators/) |

[**Structured Distillation of Web Agent Capabilities Enables Generalization**](https://arxiv.org/abs/2604.07776)

*Xing Han Lù, Siva Reddy*

</div>

This repository contains the code for the A3 framework, which uses LLMs to systematically generate synthetic web agent training data by decomposing the annotation process into three roles: **Task Designer**, **Annotator**, and **Supervisor**.

## Installation

```bash
pip install agent-as-annotators
```

Or install from source:

```bash
git clone https://github.com/McGill-NLP/agent-as-annotators.git
cd agent-as-annotators
pip install -e .
```

## Quick Start: Evaluation

### 1. Serve a model with vLLM

```bash
vllm serve --config configs/vllm/Qwen3.5-9B.yaml
```

### 2. Run evaluation

```bash
a3-eval --benchmark webarena_test --model A3-qwen3.5-9b
```

## Pipeline: Generating A3-Synth

The A3 pipeline generates synthetic training data in 5 steps:

### Step 1: Create personas
```bash
python scripts/create_personas.py
```

### Step 2: Generate task intents (via exploration)
```bash
# 2a. Run the exploration agent. Trajectories are saved as agentlab pickles
# under $AGENTLAB_EXP_ROOT/<study_dir>/<task_dir>/step_*.pkl.gz.
a3-explore

# 2b. Extract chat messages from each step pickle into a parallel JSON tree
# at outputs/chat_messages/<study_dir>/<task_dir>/step_*.json.
python scripts/extract_chat_messages.py --find-latest <exploration-model>

# 2c. For each trajectory, randomly sample N steps (default 3, skipping step 0),
# append the TASK_INTENT_PROMPT_TEMPLATE as a final user turn, and write each
# prompt as outputs/task_intents/prompts/<exploration_model>/task_<i>.step_<j>.json.
python scripts/prepare_tasks_intents_prompts.py --find-latest <exploration-model>

# 2d. Send each prepared prompt to the Task Designer LLM. Completions land in
# outputs/task_intents/completions/<exploration_model>/<task_designer_model>/.
python scripts/generate_task_intents.py \
    --exploration-model <exploration-model> \
    --model <task-designer-model>
```

#### What exactly gets passed to the Task Designer

**One exploration step, not the whole trajectory.** Each prepared prompt is the
exploration agent's own prompt/response pair *at a single sampled step*, with one
user turn appended. For the released `gemini-3-pro-preview` run that is exactly
four messages:

| # | role | content |
| - | ---- | ------- |
| 0 | `system` | the exploration agent's system prompt |
| 1 | `user` | the agent's step prompt, as a text part **plus the screenshot** of that step (`image_url`). The text holds `# Instructions`, `## Goal:` (the exploration instruction and the persona), `# Observation of current step:` (open tabs, **AXTree**, focused element), `# History of interaction with the task:`, `# Action space:` and the formatting examples |
| 2 | `assistant` | what the explorer actually produced at that step (`<thought>…</thought><action>…</action>`) |
| 3 | `user` | appended by `prepare_tasks_intents_prompts.py`: `TASK_INTENT_PROMPT_TEMPLATE.format(annotator_instructions=WEBARENA_ANNOTATOR_INSTRUCTIONS, num_intents=…)` |

So the Task Designer sees the **AXTree and screenshot of the sampled step only**.
Earlier steps reach it solely through the agent's `# History of interaction with
the task:` block, which lists *past actions and nothing else*
(`## step 0 <action>click('156')</action>` …) — no earlier observations, and no
multi-turn chat history.

#### How `exploration_step_num` is chosen

`prepare_tasks_intents_prompts.py` globs `step_*.json` in each trajectory, sorts
by step number, **drops `step_0`**, and takes a uniform random sample without
replacement:

```python
random.seed(seed)                                     # --seed, default 42
sampled = random.sample(step_files, min(num_samples, len(step_files)))
```

Each sampled step becomes one prompt file `task_<task_num>.step_<step_num>.json`,
and that `step_num` is what `scripts/create_synth_configs.py` records as
`exploration_step_num` in the A3-Synth task configs. The released run used the
defaults `--num_samples 3`, `--num_intents 2`, `--seed 42`, which is why
`exploration_step_num` in `A3-Synth` is never `0`, is capped at the exploration
budget of 20 steps, and appears at most three times per exploration trajectory.

**Known wart, kept for reproducibility.** A trajectory's *terminal* step records
no agent call, so it extracts to `messages: []` and, if sampled, yields a prompt
whose only turn is the appended instruction — the Task Designer is asked to write
tasks "based on the conversation above" with no conversation above. This affected
**381 of 4497 prompts (8.5%)** in the released run. The default behaviour is
unchanged so that generation reproduces exactly; pass `--skip-empty-steps` to
exclude those steps in new collections.

### Step 3: Create A3-Synth task configs
```bash
python scripts/create_synth_configs.py
```

### Step 4: Collect trajectories
```bash
a3-synth --benchmark a3_synth --model gemini-3-pro
```

### Step 5: Convert to training data
```bash
python scripts/convert_trajectories_to_json.py
python scripts/generate_rft_data.py
```

## Training

```bash
a3-train --config configs/train/qwen3.5-9b.json
```

Training uses SFT with FSDP for multi-GPU parallelism. See `configs/train/` for hyperparameters and `configs/accelerate/` for FSDP configuration.

## CLI Commands

| Command | Description |
|---------|-------------|
| `a3-eval` | Run evaluation on WebArena, VisualWebArena, WorkArena, MiniWoB |
| `a3-synth` | Run trajectory collection for A3-Synth |
| `a3-explore` | Run environment exploration |
| `a3-train` | Fine-tune a model with SFT |
| `a3-screen-utils` | Screen session management utilities |

## Project Structure

```
agent-as-annotators/
  agent_as_annotators/       # Core package
    cli/                     # CLI entry points (eval, synth, explore, train)
    modeling.py              # Agent model wrapper (vLLM, Gemini, OpenAI)
    prompts/                 # All prompt templates
    judge/                   # Inverted evaluation protocol (Judge module)
    benchmarks/a3_synth/     # A3-Synth benchmark registration
    exploration/             # Exploration task registration
    utils/                   # Utilities
    configs/a3_synth/        # A3-Synth task configurations
  configs/
    model_configs.json       # Model registry
    train/                   # Training hyperparameters
    vllm/                    # vLLM serving configs
    accelerate/              # FSDP configs
  scripts/                   # Data pipeline scripts
```
