# Noisy Student Self-Distillation for LLMs

Exploring whether consensus-based pseudo-labeling can expand training data for LLM reasoning tasks. Uses the RLVR (Reinforcement Learning with Verifiable Rewards) setting where rewards come from automatic verification (e.g., checking if a math answer is correct).

This project is based on a research direction suggested by [Thinking Machines Lab](https://thinkingmachines.ai/blog/call-for-community-projects/): adapting self-distillation techniques to leverage unlabeled datasets in language models.

## The idea

```
1. Train model on small labeled dataset (where we know correct answers)
2. Generate multiple answers per unlabeled problem
3. Take majority vote as pseudo-label
4. Train on this expanded dataset
5. Repeat
```

If a model generates the same answer 8/10 times, it's probably right. We use these high-confidence pseudo-labels to expand training data.

## Setup

```bash
# Create environment
uv venv
source .venv/bin/activate
uv sync

# Run notebooks locally
uv run jupyter notebook notebooks/
```

## Running experiments

Uses [Modal](https://modal.com) for GPU access. Pay-per-use, no idle costs.

```bash
# One-time auth
pip install modal
modal setup

# Test GPU works (~$0.01)
modal run modal_app.py::test_gpu_setup

# Run GRPO training on GSM8K
modal run modal_app.py::train_grpo_gsm8k
```

## Project structure

```
modal_app.py              # GPU compute entry point
src/
  data/
    gsm8k.py              # Dataset loading, answer extraction
    pseudo_label.py       # Consensus generation
  training/
    rewards.py            # Reward functions for GRPO
experiments/              # Versioned experiment scripts
notebooks/                # Learning notebooks (RL basics, TRL intro)
docs/experimental_plan.md # Experiment log and results
```

## Results

Tested on GSM8K (grade school math) with Qwen-2.5-0.5B-Instruct.

| Condition | Test Accuracy | Notes |
|-----------|---------------|-------|
| SFT Baseline (1K labeled) | 27% | Reference |
| SFT Noisy Student (1K + 4K pseudo) | 27% | Pseudo-labels don't help |
| SFT Oracle (5K real labels) | 32% | Upper bound for SFT |
| **GRPO Baseline (1K labeled)** | **33%** | Matches SFT Oracle |
| GRPO Noisy Student (1K + 4K pseudo) | 31% | Noisy labels hurt |

**Key findings:**
- GRPO on 1K labeled matches SFT on 5K real labels
- GRPO is more robust to label noise than SFT, but noisy pseudo-labels still hurt
- Best strategy: Use GRPO on clean labeled data - no pseudo-labels needed

The noisy student approach doesn't help when you have GRPO. GRPO extracts more signal from clean data than SFT can from 5x more data.

## Dependencies

- PyTorch, Transformers, TRL (HuggingFace's RL library)
- PEFT for LoRA
- Modal for GPU compute
- wandb for experiment tracking
