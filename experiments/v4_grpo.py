"""
Experiment V4: RL (GRPO) vs SFT Comparison

Research Question: Does RL learn from noisy pseudo-labels where SFT fails?

Setup (matches V1 exactly for fair comparison):
- 1K labeled GSM8K (seed=42)
- 4K unlabeled GSM8K (pseudo-labeled with K=8 consensus)
- Same model: Qwen2.5-0.5B-Instruct
- Same pseudo-label settings: temp=0.7, max_tokens=256
- Same eval settings: temp=0.1, max_tokens=256

Key difference from V1:
- V1 used SFT (cross-entropy loss on pseudo-labeled completions)
- V4 uses GRPO (RL with reward = 1 if answer matches pseudo-label)

GRPO Config (tuned via debug experiments 2024-12-25):
- temperature=0.3 (lower for coherent generations)
- num_generations=8 (enough for reward variance)
- loss_type="dr_grpo" (fixes zero-loss with binary rewards)
- learning_rate=1e-5 (faster learning)
- beta=0.04 (KL penalty)

Hypothesis: RL is more robust to label noise because it doesn't directly
fit to wrong answers - it explores via sampling and only reinforces
completions that happen to match the target.

Run with:
    modal run experiments/v4_grpo.py::run_grpo_smoke_test
    modal run experiments/v4_grpo.py::run_grpo_baseline
    modal run experiments/v4_grpo.py::run_grpo_noisy_student
"""

import modal

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.28",
    "trl>=0.8",
    "peft>=0.10",
)

app = modal.App("noisy-student-v4-grpo", image=image)
volume = modal.Volume.from_name("noisy-student-checkpoints", create_if_missing=True)


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_COMPLETION_LENGTH = 256  # Match V1 exactly for fair comparison
MAX_EVAL_TOKENS = 256  # For evaluation - matches V1 SFT
MAX_PSEUDO_LABEL_TOKENS = 256  # For pseudo-labeling - matches V1 SFT
K = 8  # Samples for pseudo-labeling (matches V1)

# GRPO training config (tuned via debug experiments)
GRPO_TEMPERATURE = 0.3  # Lower temp for coherent generations
GRPO_NUM_GENERATIONS = 8  # Enough for reward variance
GRPO_LEARNING_RATE = 1e-5  # Higher LR for faster learning
GRPO_LOSS_TYPE = "dr_grpo"  # Fixes zero-loss issue with binary rewards


# ============================================================================
# Data Loading & Utilities (same as V1 for consistency)
# ============================================================================

def load_gsm8k_splits(labeled_size: int = 1000, seed: int = 42):
    """Load GSM8K with labeled/unlabeled split."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"].shuffle(seed=seed)

    labeled = train_data.select(range(labeled_size))
    unlabeled = train_data.select(range(labeled_size, len(train_data)))

    print(f"Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}, Test: {len(dataset['test'])}")
    return labeled, unlabeled, dataset["test"]


def extract_answer(text: str) -> str | None:
    """Extract numerical answer from GSM8K format or model output."""
    import re

    # Try #### format
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')

    # Try \boxed{}
    match = re.search(r'\\boxed\{(-?[\d,]+\.?\d*)\}', text)
    if match:
        return match.group(1).replace(',', '')

    # Try "answer is X"
    match = re.search(r'answer is[:\s]*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # Last number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def format_prompt(question: str) -> str:
    """Format question as prompt."""
    # MUST match V1 exactly for fair comparison
    return f"Solve this math problem step by step. End with #### followed by the numerical answer.\n\nProblem: {question}\n\nSolution:"


def extract_question_from_prompt(prompt: str) -> str:
    """Extract original question from formatted prompt."""
    if "Problem:" in prompt and "Solution:" in prompt:
        try:
            start = prompt.index("Problem:") + len("Problem:")
            end = prompt.index("\n\nSolution:")
            return prompt[start:end].strip()
        except ValueError:
            pass
    return prompt.strip()


def answers_match(a1: str | None, a2: str | None) -> bool:
    """Check if two answers match numerically."""
    if a1 is None or a2 is None:
        return False
    try:
        return abs(float(a1) - float(a2)) < 1e-6
    except ValueError:
        return a1.strip() == a2.strip()


# ============================================================================
# Reward Functions
# ============================================================================

def make_reward_fn(answer_lookup: dict, debug: bool = False):
    """
    Create a reward function for GRPO.

    Args:
        answer_lookup: Dict mapping question -> answer (ground truth or pseudo-label)
        debug: Print debug info

    Returns:
        Reward function compatible with GRPOTrainer
    """
    stats = {"hits": 0, "misses": 0, "correct": 0, "total": 0}
    call_count = [0]
    completion_lengths = []  # Track for debugging cutoff issues

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        call_count[0] += 1
        rewards = []

        for completion, prompt in zip(completions, prompts):
            # Track completion length
            completion_lengths.append(len(completion))

            # Extract question from prompt
            question = extract_question_from_prompt(prompt)

            # Get target answer (ground truth or pseudo-label)
            target_answer = answer_lookup.get(question)

            if target_answer is None:
                stats["misses"] += 1
                if debug and stats["misses"] <= 5:
                    print(f"WARNING: Unknown question (first 50 chars): {question[:50]}...")
                rewards.append(0.0)
                continue

            stats["hits"] += 1
            stats["total"] += 1

            # Extract answer from model output
            extracted = extract_answer(completion)

            # Check for cutoff (no answer found and completion is long)
            if extracted is None and len(completion) > 400:
                if debug and call_count[0] <= 10:
                    print(f"CUTOFF? len={len(completion)}, ends with: ...{completion[-100:]}")

            # Compare
            match = answers_match(extracted, target_answer)
            if match:
                stats["correct"] += 1

            rewards.append(1.0 if match else 0.0)

            # Debug logging
            if debug and call_count[0] % 50 == 0 and len(rewards) == 1:
                print(f"\n--- Reward Debug (call #{call_count[0]}) ---")
                print(f"Completion length: {len(completion)} chars")
                print(f"Extracted: {extracted}, Target: {target_answer}, Match: {match}")
                print(f"Stats: {stats['correct']}/{stats['total']} correct")
                avg_len = sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0
                print(f"Avg completion length: {avg_len:.0f} chars")

        return rewards

    reward_fn.stats = stats
    reward_fn.completion_lengths = completion_lengths
    return reward_fn


# ============================================================================
# Evaluation Helper
# ============================================================================

def evaluate(model, tokenizer, test_data, name, sample_size=500):
    """Evaluate model on test set with proper token limits."""
    import torch

    model.eval()
    correct = 0
    total_length = 0
    cutoff_count = 0

    sample = test_data.shuffle(seed=42).select(range(min(sample_size, len(test_data))))

    for i, example in enumerate(sample):
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_EVAL_TOKENS,  # Match V1 for fair comparison
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):]
        total_length += len(completion)

        pred = extract_answer(completion)
        true = extract_answer(example["answer"])

        # Track potential cutoffs
        if pred is None and len(completion) > 400:
            cutoff_count += 1

        if answers_match(pred, true):
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"  Eval {i + 1}/{sample_size}: {correct}/{i+1} ({correct/(i+1):.1%})", flush=True)

    acc = correct / len(sample)
    avg_length = total_length / len(sample)

    print(f"{name}: {acc:.1%} ({correct}/{len(sample)})")
    print(f"  Avg completion length: {avg_length:.0f} chars, Potential cutoffs: {cutoff_count}")

    return acc


# ============================================================================
# Smoke Test
# ============================================================================

@app.function(gpu="L4", timeout=60 * 30)  # 30 min
def run_grpo_smoke_test():
    """
    Quick test to verify GRPO training works with our setup.
    Uses tiny data to complete fast.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print("=" * 60, flush=True)
    print("GRPO SMOKE TEST", flush=True)
    print("=" * 60, flush=True)

    # Load tiny subset
    labeled, _, test = load_gsm8k_splits(labeled_size=50)

    # Load model
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Create prompt dataset for GRPO
    prompts = [format_prompt(ex["question"]) for ex in labeled]
    train_dataset = Dataset.from_dict({"prompt": prompts})

    # Build answer lookup
    answer_lookup = {}
    for ex in labeled:
        question = ex["question"].strip()
        answer = extract_answer(ex["answer"])
        if answer:
            answer_lookup[question] = answer

    print(f"Answer lookup size: {len(answer_lookup)}", flush=True)

    # Create reward function
    reward_fn = make_reward_fn(answer_lookup, debug=True)

    # GRPO config - small for smoke test (uses tuned settings)
    config = GRPOConfig(
        output_dir="./grpo_smoke_output",
        num_generations=GRPO_NUM_GENERATIONS,
        learning_rate=GRPO_LEARNING_RATE,
        per_device_train_batch_size=GRPO_NUM_GENERATIONS,  # Must match num_generations
        max_steps=20,  # Just 20 steps for smoke test
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=GRPO_TEMPERATURE,
        loss_type=GRPO_LOSS_TYPE,
        beta=0.04,
        logging_steps=5,
        report_to="none",
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
    )

    # Train
    print("\nStarting GRPO training (20 steps)...", flush=True)
    trainer.train()

    # Print stats
    print(f"\n{'=' * 60}", flush=True)
    print("Reward Function Stats:", flush=True)
    print(f"  Hits: {reward_fn.stats['hits']}", flush=True)
    print(f"  Misses: {reward_fn.stats['misses']}", flush=True)
    print(f"  Correct: {reward_fn.stats['correct']}/{reward_fn.stats['total']}", flush=True)

    if reward_fn.completion_lengths:
        avg_len = sum(reward_fn.completion_lengths) / len(reward_fn.completion_lengths)
        max_len = max(reward_fn.completion_lengths)
        min_len = min(reward_fn.completion_lengths)
        print(f"  Completion lengths: avg={avg_len:.0f}, min={min_len}, max={max_len}", flush=True)

    # Quick eval
    print("\nQuick evaluation (20 examples)...", flush=True)
    acc = evaluate(model, tokenizer, test, "Smoke Test", sample_size=20)

    print(f"\n{'=' * 60}", flush=True)
    print("SMOKE TEST COMPLETE", flush=True)
    print(f"{'=' * 60}", flush=True)

    return {
        "status": "passed",
        "accuracy": acc,
        "reward_stats": reward_fn.stats,
    }


# ============================================================================
# V4 Baseline: GRPO on 1K Labeled
# ============================================================================

@app.function(gpu="A10G", timeout=60 * 60 * 10, volumes={"/checkpoints": volume})  # 10 hours (A10G is ~40% faster)
def run_grpo_baseline(
    labeled_size: int = 1000,
    max_steps: int = 500,
    eval_size: int = 500,
):
    """
    GRPO training on labeled data only.
    This is the RL equivalent of V1's SFT baseline.
    """
    import torch
    import json
    import time
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print("=" * 60, flush=True)
    print(f"V4 GRPO BASELINE (labeled_size={labeled_size})", flush=True)
    print("=" * 60, flush=True)

    # Load data
    labeled, _, test = load_gsm8k_splits(labeled_size=labeled_size)

    # Load model
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Create prompt dataset
    prompts = [format_prompt(ex["question"]) for ex in labeled]
    train_dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Train dataset: {len(train_dataset)} examples", flush=True)

    # Build answer lookup (ground truth)
    answer_lookup = {}
    for ex in labeled:
        question = ex["question"].strip()
        answer = extract_answer(ex["answer"])
        if answer:
            answer_lookup[question] = answer

    print(f"Answer lookup: {len(answer_lookup)} questions", flush=True)

    # Create reward function
    reward_fn = make_reward_fn(answer_lookup, debug=True)

    # GRPO config (tuned via debug experiments)
    config = GRPOConfig(
        output_dir="./grpo_baseline_output",
        num_generations=GRPO_NUM_GENERATIONS,
        learning_rate=GRPO_LEARNING_RATE,
        per_device_train_batch_size=GRPO_NUM_GENERATIONS,  # Must match num_generations
        max_steps=max_steps,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=GRPO_TEMPERATURE,
        loss_type=GRPO_LOSS_TYPE,
        beta=0.04,
        logging_steps=50,
        report_to="none",
    )

    # Train
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
    )

    print(f"\nStarting GRPO training ({max_steps} steps)...", flush=True)
    trainer.train()

    # Stats
    print(f"\n{'=' * 60}", flush=True)
    print("Training Stats:", flush=True)
    print(f"  Reward hits: {reward_fn.stats['hits']}", flush=True)
    print(f"  Reward misses: {reward_fn.stats['misses']}", flush=True)
    if reward_fn.stats['total'] > 0:
        train_acc = reward_fn.stats['correct'] / reward_fn.stats['total']
        print(f"  Training accuracy: {train_acc:.1%}", flush=True)

    if reward_fn.completion_lengths:
        avg_len = sum(reward_fn.completion_lengths) / len(reward_fn.completion_lengths)
        print(f"  Avg completion length: {avg_len:.0f} chars", flush=True)

    # Evaluate
    print(f"\nEvaluating on test set ({eval_size} examples)...", flush=True)
    test_acc = evaluate(model, tokenizer, test, "GRPO Baseline", sample_size=eval_size)

    print(f"\n{'=' * 60}", flush=True)
    print("V4 GRPO BASELINE RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Test Accuracy: {test_acc:.1%}", flush=True)
    print(f"\nComparison to V1 SFT:", flush=True)
    print(f"  V1 SFT Baseline: 27.0%", flush=True)
    print(f"  V4 GRPO Baseline: {test_acc:.1%}", flush=True)

    run_id = f"v4_baseline_{int(time.time())}"
    results = {
        "run_id": run_id,
        "test_accuracy": test_acc,
        "train_accuracy": reward_fn.stats['correct'] / max(reward_fn.stats['total'], 1),
        "reward_stats": reward_fn.stats,
        "config": {
            "labeled_size": labeled_size,
            "max_steps": max_steps,
            "eval_size": eval_size,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "max_eval_tokens": MAX_EVAL_TOKENS,
            "grpo_temperature": GRPO_TEMPERATURE,
            "grpo_num_generations": GRPO_NUM_GENERATIONS,
            "grpo_loss_type": GRPO_LOSS_TYPE,
            "grpo_learning_rate": GRPO_LEARNING_RATE,
        }
    }

    # Save results
    results_path = f"/checkpoints/{run_id}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}", flush=True)
    volume.commit()

    return results


# ============================================================================
# V4 Noisy Student: GRPO on 1K Labeled + 4K Pseudo-labeled
# ============================================================================

@app.function(gpu="A10G", timeout=60 * 60 * 20, volumes={"/checkpoints": volume})  # A10G ~40% faster, 20h should be plenty
def run_grpo_noisy_student(
    labeled_size: int = 1000,
    unlabeled_size: int = 4000,
    max_steps: int = 2500,  # Match V1: 5K examples * 2 epochs / batch_size=4
    eval_size: int = 500,
):
    """
    GRPO training on labeled + pseudo-labeled data.
    This is the RL equivalent of V1's noisy student.

    Key difference from SFT:
    - SFT fits to the pseudo-labeled completion directly
    - GRPO rewards matching the pseudo-label answer, but explores via sampling
    """
    import torch
    import json
    import time
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from collections import Counter

    print("=" * 60, flush=True)
    print(f"V4 GRPO NOISY STUDENT", flush=True)
    print(f"  Labeled: {labeled_size}, Unlabeled: {unlabeled_size}", flush=True)
    print("=" * 60, flush=True)

    # Load data
    labeled, unlabeled_full, test = load_gsm8k_splits(labeled_size=labeled_size)
    unlabeled = unlabeled_full.shuffle(seed=42).select(range(unlabeled_size))

    # Load model
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 1: Generate pseudo-labels ----
    print(f"\n{'=' * 40}", flush=True)
    print(f"Step 1: Pseudo-labeling {unlabeled_size} examples (K={K})", flush=True)
    print("=" * 40, flush=True)

    pseudo_labels = {}  # question -> pseudo-label answer
    correct_pseudo = 0

    model.eval()
    for i, example in enumerate(unlabeled):
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate K responses (matches V1 exactly)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_PSEUDO_LABEL_TOKENS,  # 256 tokens (matches V1)
                temperature=0.7,
                do_sample=True,
                num_return_sequences=K,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        answers = [extract_answer(r) for r in responses]

        # Majority vote
        valid = [a for a in answers if a is not None]
        if valid:
            consensus = Counter(valid).most_common(1)[0][0]
            question = example["question"].strip()
            pseudo_labels[question] = consensus

            # Check accuracy (we secretly have ground truth)
            true_ans = extract_answer(example["answer"])
            if answers_match(consensus, true_ans):
                correct_pseudo += 1

        if (i + 1) % 500 == 0:
            acc_so_far = correct_pseudo / (i + 1) if i > 0 else 0
            print(f"  Pseudo-labeled {i + 1}/{unlabeled_size} ({acc_so_far:.1%} accurate)", flush=True)

    pseudo_acc = correct_pseudo / len(pseudo_labels) if pseudo_labels else 0
    print(f"\nPseudo-labeling complete:", flush=True)
    print(f"  Generated: {len(pseudo_labels)} pseudo-labels", flush=True)
    print(f"  Accuracy: {pseudo_acc:.1%} (vs ground truth)", flush=True)

    # Save pseudo-labels to volume for debugging/reuse
    run_id = f"v4_noisy_{int(time.time())}"
    pseudo_labels_path = f"/checkpoints/{run_id}_pseudo_labels.json"
    with open(pseudo_labels_path, "w") as f:
        json.dump({
            "pseudo_labels": pseudo_labels,
            "pseudo_label_accuracy": pseudo_acc,
            "pseudo_label_count": len(pseudo_labels),
            "config": {
                "labeled_size": labeled_size,
                "unlabeled_size": unlabeled_size,
                "K": K,
                "max_completion_length": MAX_COMPLETION_LENGTH,
            }
        }, f, indent=2)
    print(f"  Saved pseudo-labels to {pseudo_labels_path}", flush=True)
    volume.commit()

    # ---- Step 2: Build combined answer lookup ----
    print(f"\n{'=' * 40}", flush=True)
    print("Step 2: Building combined answer lookup", flush=True)
    print("=" * 40, flush=True)

    # Ground truth for labeled examples
    answer_lookup = {}
    for ex in labeled:
        question = ex["question"].strip()
        answer = extract_answer(ex["answer"])
        if answer:
            answer_lookup[question] = answer

    # Add pseudo-labels for unlabeled examples
    answer_lookup.update(pseudo_labels)

    print(f"  Labeled answers: {labeled_size}", flush=True)
    print(f"  Pseudo-labels: {len(pseudo_labels)}", flush=True)
    print(f"  Total: {len(answer_lookup)}", flush=True)

    # ---- Step 3: Create training dataset ----
    # Combine labeled + unlabeled prompts
    all_prompts = []
    for ex in labeled:
        all_prompts.append(format_prompt(ex["question"]))
    for ex in unlabeled:
        if ex["question"].strip() in pseudo_labels:
            all_prompts.append(format_prompt(ex["question"]))

    train_dataset = Dataset.from_dict({"prompt": all_prompts})
    print(f"  Training prompts: {len(train_dataset)}", flush=True)

    # ---- Step 4: GRPO Training ----
    print(f"\n{'=' * 40}", flush=True)
    print(f"Step 3: GRPO Training ({max_steps} steps)", flush=True)
    print("=" * 40, flush=True)

    # Reload fresh model for fair comparison
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Create reward function with combined lookup
    reward_fn = make_reward_fn(answer_lookup, debug=True)

    # GRPO config (tuned via debug experiments)
    config = GRPOConfig(
        output_dir="./grpo_noisy_student_output",
        num_generations=GRPO_NUM_GENERATIONS,
        learning_rate=GRPO_LEARNING_RATE,
        per_device_train_batch_size=GRPO_NUM_GENERATIONS,  # Must match num_generations
        max_steps=max_steps,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=GRPO_TEMPERATURE,
        loss_type=GRPO_LOSS_TYPE,
        beta=0.04,
        logging_steps=50,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
    )

    trainer.train()

    # ---- Step 5: Evaluate ----
    print(f"\n{'=' * 40}", flush=True)
    print(f"Step 4: Evaluation ({eval_size} examples)", flush=True)
    print("=" * 40, flush=True)

    if reward_fn.stats['total'] > 0:
        train_acc = reward_fn.stats['correct'] / reward_fn.stats['total']
        print(f"Training accuracy: {train_acc:.1%}", flush=True)

    test_acc = evaluate(model, tokenizer, test, "GRPO Noisy Student", sample_size=eval_size)

    # ---- Results ----
    print(f"\n{'=' * 60}", flush=True)
    print("V4 GRPO NOISY STUDENT RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Pseudo-label accuracy: {pseudo_acc:.1%}", flush=True)
    print(f"Test accuracy: {test_acc:.1%}", flush=True)
    print(f"\nComparison to V1 SFT:", flush=True)
    print(f"  V1 SFT Baseline:       27.0%", flush=True)
    print(f"  V1 SFT Noisy Student:  26.8%", flush=True)
    print(f"  V4 GRPO Noisy Student: {test_acc:.1%}", flush=True)

    results = {
        "run_id": run_id,
        "test_accuracy": test_acc,
        "pseudo_label_accuracy": pseudo_acc,
        "pseudo_label_count": len(pseudo_labels),
        "train_accuracy": reward_fn.stats['correct'] / max(reward_fn.stats['total'], 1),
        "reward_stats": reward_fn.stats,
        "config": {
            "labeled_size": labeled_size,
            "unlabeled_size": unlabeled_size,
            "max_steps": max_steps,
            "eval_size": eval_size,
            "K": K,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "max_eval_tokens": MAX_EVAL_TOKENS,
            "max_pseudo_label_tokens": MAX_PSEUDO_LABEL_TOKENS,
            "grpo_temperature": GRPO_TEMPERATURE,
            "grpo_num_generations": GRPO_NUM_GENERATIONS,
            "grpo_loss_type": GRPO_LOSS_TYPE,
            "grpo_learning_rate": GRPO_LEARNING_RATE,
        }
    }

    # Save final results
    results_path = f"/checkpoints/{run_id}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}", flush=True)
    volume.commit()

    return results


# ============================================================================
# Entry point
# ============================================================================

@app.local_entrypoint()
def main():
    print("V4 GRPO Experiment Commands:")
    print("  modal run experiments/v4_grpo.py::run_grpo_smoke_test")
    print("  modal run experiments/v4_grpo.py::run_grpo_baseline")
    print("  modal run experiments/v4_grpo.py::run_grpo_noisy_student")
