<div align="center">

# Agent-as-Annotators (A3)

| [**💾 Code**](https://github.com/McGill-NLP/agent-as-annotators) | [**📄 Paper**](https://arxiv.org/abs/TODO) | [**🌐 Website**](https://agent-as-annotators.github.io) |
| :--: | :--: | :--: |
| [**🤗 Dataset**](https://huggingface.co/datasets/McGill-NLP/A3-Synth) | [**🤖 Model**](https://huggingface.co/McGill-NLP/A3-Qwen3.5-9B) | [**📦 PyPI**](https://pypi.org/project/agent-as-annotators/) |

[**Structured Distillation of Web Agent Capabilities Enables Generalization**](https://arxiv.org/abs/TODO)

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
a3-explore
python scripts/generate_task_intents.py
```

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
