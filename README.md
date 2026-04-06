<div align="center">

# Agent-as-Annotators (A3)

| [**💾 Code**](https://github.com/McGill-NLP/agent-as-annotators) | [**📄 Paper**](https://arxiv.org/abs/TODO) | [**🌐 Website**](https://agent-as-annotators.github.io) |
| :--: | :--: | :--: |
| [**🤗 Dataset**](https://huggingface.co/datasets/McGill-NLP/A3-Synth) | [**🤖 Model**](https://huggingface.co/McGill-NLP/A3-Qwen3.5-9B) | |

[**Structured Distillation of Web Agent Capabilities Enables Generalization**](https://arxiv.org/abs/TODO)

*Xing Han Lu, Siva Reddy*

</div>

This repository contains the code for the A3 framework, which uses LLMs to systematically generate synthetic web agent training data by decomposing the annotation process into three roles: **Task Designer**, **Annotator**, and **Supervisor**.

## Installation

```bash
pip install -e .
```

## Quick Start: Evaluation

### 1. Serve a model with vLLM

```bash
vllm serve --config configs/vllm/Qwen3.5-9B.yaml
```

### 2. Run evaluation

```bash
# WebArena
python run_webarena.py --benchmark webarena_test --model A3-qwen3.5-9b

# VisualWebArena
python run_webarena.py --benchmark visualwebarena_test --model A3-qwen3.5-9b

# WorkArena L1
python run_webarena.py --benchmark workarena_l1 --model A3-qwen3.5-9b
```

## Pipeline: Generating A3-Synth

The A3 pipeline generates synthetic training data in 5 steps:

### Step 1: Create personas
```bash
python scripts/create_personas.py
```

### Step 2: Generate task intents (via exploration)
```bash
python run_exploration.py
python scripts/generate_task_intents.py
```

### Step 3: Create WebSynth task configs
```bash
python scripts/create_websynth_configs.py
```

### Step 4: Collect trajectories
```bash
python run_websynth.py
```

### Step 5: Convert to training data
```bash
python scripts/convert_trajectories_to_json.py
python scripts/generate_rft_data.py
```

## Training

```bash
python train.py --config configs/train/qwen3.5-9b.json
```

Training uses SFT with FSDP for multi-GPU parallelism. See `configs/train/` for hyperparameters and `configs/accelerate/` for FSDP configuration.

## Project Structure

```
agent-as-annotators/
  llm_annotators/           # Core package
    modeling.py              # Agent model wrapper (vLLM, Gemini, OpenAI)
    prompts/                 # All prompt templates (exploration, task creation, annotation)
    judge/                   # Inverted evaluation protocol (Judge module)
    benchmarks/websynth/     # WebSynth benchmark registration
    exploration/             # Exploration task registration
    utils/                   # Utilities
    configs/websynth/        # WebSynth task configurations
  configs/
    model_configs.json       # Model registry (base models + A3 fine-tuned)
    train/                   # Training hyperparameters
    vllm/                    # vLLM serving configs
    accelerate/              # FSDP configs
  scripts/                   # Data pipeline scripts
  train.py                   # SFT training script
  run_webarena.py            # Evaluation entry point
  run_websynth.py            # Trajectory collection
  run_exploration.py         # Environment exploration
```