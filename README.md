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

**What gets passed to the Task Designer:** the *full chat-message history* of a
single exploration step (system prompt, goal, every prior assistant action and
observation up to that step) with one extra user turn appended that contains
`TASK_INTENT_PROMPT_TEMPLATE`. The step is selected by uniform random sampling
of `step_*.json` files in the trajectory, after dropping `step_0` (the initial
observation, which has no agent actions yet). The number of samples per
trajectory is controlled by `--num_samples` (default `3`); the number of intents
requested per prompt is `--num_intents` (default `2`). Sampling is seeded by
`--seed` (default `42`) for reproducibility.

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
