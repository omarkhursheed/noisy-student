# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Noisy Student Self-Distillation for LLMs** in the RLVR (Reinforcement Learning with Verifiable Rewards) setting. This is a learning-focused project to gain practical RL experience while exploring a novel research direction.

### What is RLVR?
Reinforcement Learning with Verifiable Rewards is a simpler form of RLHF where instead of training a reward model, you use automatic verification (e.g., checking if a math answer is correct). This makes it accessible for learning RL because:
- No need for human preference data
- Clear, binary reward signals (correct/incorrect)
- Easy to debug and understand what's happening

### Core Algorithm (Plain English)

```
1. Train model with RL on small labeled dataset (where we know correct answers)
2. Use that model to generate many answers for unlabeled problems
3. Take the most common answer as the "pseudo-label" (consensus/majority vote)
4. Train with RL on this larger pseudo-labeled dataset
5. Repeat: relabel with better model, train again
```

The intuition: If the model generates the same answer 8/10 times, it's probably right. We use these high-confidence pseudo-labels to expand our training data.

## Development Commands

```bash
# Environment setup with uv
uv venv
source .venv/bin/activate
uv sync  # installs from pyproject.toml

# Jupyter for exploration (local, no GPU needed)
uv run jupyter notebook notebooks/

# Tests
uv run pytest tests/ -v
```

## Modal (GPU Compute)

We use Modal for GPU access. Serverless = pay only for GPU time used.

```bash
# One-time setup
pip install modal
modal setup  # authenticates with Modal

# Test GPU access (costs ~$0.01)
modal run modal_app.py::test_gpu_setup

# Run training on cloud GPU
modal run modal_app.py::train_rl

# Deploy as persistent endpoint
modal deploy modal_app.py
```

### GPU Cost Reference
- **L4 (24GB): ~$0.80/hr** - Default choice, good for 1-3B models
- T4 (16GB): ~$0.59/hr - Testing only
- A10G (24GB): ~$1.10/hr - Faster training
- A100 (40GB): ~$2.78/hr - For 7B+ models

**Cost efficiency rule**: Develop and debug locally or with L4. Only upgrade GPU when hitting memory limits or when running final experiments.

### Modal Secrets (for wandb, HuggingFace)
```bash
modal secret create wandb WANDB_API_KEY=your_key
modal secret create huggingface HF_TOKEN=your_token
```

## RL Concepts Reference

### Key Terms
- **Policy**: The model's strategy for generating text (i.e., the LLM itself)
- **Reward**: Score for an output (for us: 1 if correct, 0 if wrong)
- **Episode**: One complete generation (prompt -> full response)
- **Policy Gradient**: Update model weights to make high-reward outputs more likely

### RL Algorithms We'll Use
- **REINFORCE**: Simplest policy gradient. Good starting point.
- **PPO (Proximal Policy Optimization)**: More stable, industry standard for RLHF
- **GRPO (Group Relative Policy Optimization)**: Newer, designed for LLMs, used by DeepSeek

### The `trl` Library
HuggingFace's `trl` handles the RL training loop. Key classes:
- `PPOTrainer`: Handles PPO training with a reward function
- `GRPOTrainer`: For GRPO-style training
- `SFTTrainer`: For supervised fine-tuning (distillation variant)

## Architecture

```
noisy_student/
├── modal_app.py          # GPU compute entry point
├── configs/              # YAML configs for experiments
├── src/
│   ├── models/           # Model loading, LoRA setup
│   ├── training/
│   │   ├── rl.py         # RL training loops
│   │   ├── sft.py        # Supervised fine-tuning
│   │   └── rewards.py    # Answer verification (the "verifiable" in RLVR)
│   ├── data/
│   │   ├── datasets.py   # Load GSM8K, MATH, etc.
│   │   ├── pseudo_label.py  # Generate consensus labels
│   │   └── sampling.py   # Sample multiple responses
│   └── eval/             # Evaluation metrics
├── notebooks/            # Learning and exploration
│   ├── 01_rl_basics.ipynb
│   ├── 02_trl_intro.ipynb
│   └── 03_pseudo_labeling.ipynb
└── tests/
```

## Datasets

- **GSM8K**: Grade school math, 7.5K problems. Start here - simpler.
- **MATH**: Competition math, harder. Use after GSM8K works.

Both have train/test splits with ground truth answers for verification.

## Learning Path

Suggested order for building this project:

1. **Notebook: RL basics** - Understand policy gradients on a toy problem
2. **Notebook: trl intro** - Run basic PPO on a small model with GSM8K
3. **Build reward function** - Parse math answers, verify correctness
4. **Build pseudo-labeling** - Sample K responses, find consensus
5. **First iteration** - RL on labeled -> pseudo-label -> RL on pseudo-labeled
6. **Iterate and measure** - Track metrics, run multiple iterations

## Key Resources

### Papers
- Original Noisy Student (images): arxiv.org/abs/1911.04252
- TTRL (test-time RL): arxiv.org/abs/2504.16084
- Self-training reasoning models: arxiv.org/abs/2505.21444

### Tutorials
- HuggingFace TRL docs: huggingface.co/docs/trl
- Spinning Up in Deep RL (OpenAI): spinningup.openai.com
- RLHF blog post: huggingface.co/blog/rlhf

## Testing Principles

**Smoke tests and quick tests should be realistic:**
- Reduce sample COUNT (e.g., 10 examples instead of 1000)
- Do NOT reduce quality-affecting parameters (e.g., max_tokens, temperature, K)
- Tests should exercise the same code paths as the real experiment
- If a test passes but uses different parameters, it doesn't actually validate the real behavior

## Modal Timeouts

**ALWAYS use generous timeouts for Modal functions:**
- Pseudo-labeling 4K examples with K=8: minimum 3 hours
- Full training + eval experiments: minimum 6-8 hours
- Default to `timeout=60 * 60 * 8` (8 hours) for any experiment with pseudo-labeling
- Better to have unused time than to lose progress to timeout
- Multiple experiments have been killed by timeouts - BE CONSERVATIVE

## Experiment Tracking Protocol

**IMPORTANT: After every experiment run, update the following:**

1. **docs/experimental_plan.md** - Update the Log table with:
   - Date, experiment name, key results, next step

2. **results/** - Save structured results:
   - `results/runs/{experiment_id}.json` - Full metrics and config
   - `results/checkpoints/` - Model checkpoints (on Modal volume)

3. **Decision points** - Before proceeding to next experiment, clearly state:
   - What question did this experiment answer?
   - What did we learn?
   - What's the minimal next step?

### Results Directory Structure
```
results/
├── runs/                    # JSON files per experiment
│   ├── baseline_v1_001.json
│   ├── pseudo_label_check_001.json
│   └── ...
├── checkpoints/             # Model checkpoints (Modal volume)
└── summary.csv              # Quick reference of all runs
```

### Experiment Naming Convention
`{experiment_type}_{version}_{run_id}`
- Example: `baseline_v1_001`, `pseudo_label_k8_002`, `full_v1_003`

## Development Notes

- Start with Qwen-2.5-0.5B-Instruct for fast iteration, scale up when needed
- Use LoRA (Low-Rank Adaptation) to train efficiently - only ~1% of parameters
- Log everything to Weights & Biases for debugging
- Run notebooks locally (no GPU needed for learning notebooks)
- Use Modal for actual training runs
