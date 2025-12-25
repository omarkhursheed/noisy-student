"""
Experiment V1: Simplest Noisy Student Setup

Data:
- 1K labeled GSM8K (use real labels)
- 6.5K unlabeled GSM8K (pseudo-label with consensus)
- 1.3K test (evaluation)

Method:
- SFT training (not RL)
- K=4 samples for consensus
- 1 iteration (no re-pseudo-labeling)

Run with:
    modal run experiments/v1_simple.py::run_baseline
    modal run experiments/v1_simple.py::run_pseudo_label_check
    modal run experiments/v1_simple.py::run_full_v1
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

app = modal.App("noisy-student-v1", image=image)

# Create a volume to persist checkpoints between runs
volume = modal.Volume.from_name("noisy-student-checkpoints", create_if_missing=True)


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_gsm8k_splits(labeled_size: int = 1000, seed: int = 42):
    """Load GSM8K with labeled/unlabeled split."""
    from datasets import load_dataset
    import re

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
    return f"Solve this math problem step by step. End with #### followed by the numerical answer.\n\nProblem: {question}\n\nSolution:"


def answers_match(a1: str | None, a2: str | None) -> bool:
    """Check if two answers match numerically."""
    if a1 is None or a2 is None:
        return False
    try:
        return abs(float(a1) - float(a2)) < 1e-6
    except ValueError:
        return a1.strip() == a2.strip()


# ============================================================================
# Pre-flight Check 1: Baseline Accuracy
# ============================================================================

@app.function(gpu="L4", timeout=60 * 30)
def run_baseline():
    """
    Train on 1K labeled, evaluate on test set.
    This tells us if 1K is enough to get a reasonable model.

    Target: 30-60% accuracy (if <20%, 1K is too small)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    print("=" * 60)
    print("V1 BASELINE: Train on 1K labeled, evaluate on test")
    print("=" * 60)

    # Load data
    labeled, _, test = load_gsm8k_splits(labeled_size=1000)

    # Prepare training data
    def format_example(example):
        prompt = format_prompt(example["question"])
        return {"text": prompt + " " + example["answer"]}

    train_dataset = labeled.map(format_example)

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # SFT config
    config = SFTConfig(
        output_dir="./baseline_output",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nTraining on 1K labeled...")
    trainer.train()

    # Evaluate
    print("\nEvaluating on test set...", flush=True)
    model.eval()
    correct = 0
    total = 0

    # Sample 100 for faster eval (was 200, but too slow)
    test_sample = test.shuffle(seed=42).select(range(min(100, len(test))))

    for i, example in enumerate(test_sample):
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_answer = extract_answer(response)
        true_answer = extract_answer(example["answer"])

        if answers_match(pred_answer, true_answer):
            correct += 1
        total += 1

        if total <= 3:
            print(f"\n--- Example {total} ---", flush=True)
            print(f"Q: {example['question'][:100]}...", flush=True)
            print(f"Pred: {pred_answer}, True: {true_answer}, Match: {answers_match(pred_answer, true_answer)}", flush=True)

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(test_sample)} - Running accuracy: {correct}/{total} ({correct/total:.1%})", flush=True)

    accuracy = correct / total
    print(f"\n{'=' * 60}")
    print(f"BASELINE ACCURACY: {accuracy:.1%} ({correct}/{total})")
    print(f"{'=' * 60}")

    if accuracy < 0.2:
        print("WARNING: Accuracy < 20%. Consider using more labeled data or larger model.")
    elif accuracy > 0.7:
        print("NOTE: Accuracy > 70%. Task might be too easy to see improvement.")
    else:
        print("Accuracy in target range (20-70%). Good to proceed.")

    return {"accuracy": accuracy, "correct": correct, "total": total}


# ============================================================================
# Pre-flight Check 2: Pseudo-label Quality
# ============================================================================

@app.function(gpu="L4", timeout=60 * 60 * 2)  # 2 hours (K=8 is slow)
def run_pseudo_label_check():
    """
    Generate pseudo-labels for a sample of unlabeled data.
    Check accuracy against ground truth (which we secretly have).

    Target: >65% pseudo-label accuracy
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import Counter

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    K = 8  # Samples per prompt (was 4, trying 8 for better consensus)
    SAMPLE_SIZE = 100  # Check 100 examples

    print("=" * 60, flush=True)
    print(f"PSEUDO-LABEL CHECK: K={K} samples, {SAMPLE_SIZE} examples", flush=True)
    print("=" * 60, flush=True)

    # Load data
    _, unlabeled, _ = load_gsm8k_splits(labeled_size=1000)

    # Sample for evaluation
    sample = unlabeled.shuffle(seed=42).select(range(SAMPLE_SIZE))

    # Load model (base model, not fine-tuned)
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Generate pseudo-labels
    results = []

    for i, example in enumerate(sample):
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate K responses
        answers = []
        for _ in range(K):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = extract_answer(response)
            answers.append(answer)

        # Majority vote
        valid_answers = [a for a in answers if a is not None]
        if valid_answers:
            counter = Counter(valid_answers)
            consensus, count = counter.most_common(1)[0]
            confidence = count / K
        else:
            consensus = None
            confidence = 0.0

        # Check against ground truth
        true_answer = extract_answer(example["answer"])
        is_correct = answers_match(consensus, true_answer)

        results.append({
            "consensus": consensus,
            "confidence": confidence,
            "true_answer": true_answer,
            "is_correct": is_correct,
            "all_answers": answers,
        })

        if (i + 1) % 10 == 0:
            current_acc = len([r for r in results if r["is_correct"]]) / len(results) if results else 0
            print(f"Processed {i + 1}/{SAMPLE_SIZE} - Running accuracy: {current_acc:.1%}", flush=True)

    # Compute metrics
    has_consensus = [r for r in results if r["consensus"] is not None]
    correct = [r for r in has_consensus if r["is_correct"]]

    coverage = len(has_consensus) / len(results)
    accuracy = len(correct) / len(has_consensus) if has_consensus else 0

    # Accuracy by confidence bucket
    high_conf = [r for r in has_consensus if r["confidence"] >= 0.75]
    high_conf_correct = [r for r in high_conf if r["is_correct"]]
    high_conf_acc = len(high_conf_correct) / len(high_conf) if high_conf else 0

    print(f"\n{'=' * 60}")
    print(f"PSEUDO-LABEL QUALITY RESULTS")
    print(f"{'=' * 60}")
    print(f"Coverage (has consensus): {coverage:.1%}")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"High-confidence (>=75%) accuracy: {high_conf_acc:.1%} (n={len(high_conf)})")

    if accuracy < 0.5:
        print("\nWARNING: Accuracy < 50%. Pseudo-labels unreliable.")
        print("Consider: more samples (K), stronger model, or easier task.")
    elif accuracy > 0.8:
        print("\nGood: Accuracy > 80%. Pseudo-labels are reliable.")
    else:
        print(f"\nAccuracy {accuracy:.1%} is reasonable. Proceed with caution.")

    # Show some examples
    print("\n--- Sample Results ---")
    for r in results[:5]:
        status = "CORRECT" if r["is_correct"] else "WRONG"
        print(f"  Consensus: {r['consensus']}, True: {r['true_answer']}, Conf: {r['confidence']:.0%} [{status}]")
        print(f"    All answers: {r['all_answers']}")

    return {
        "coverage": coverage,
        "accuracy": accuracy,
        "high_conf_accuracy": high_conf_acc,
        "sample_size": SAMPLE_SIZE,
        "K": K,
    }


# ============================================================================
# Full V1 Experiment
# ============================================================================

@app.function(gpu="L4", timeout=60 * 60 * 8, volumes={"/checkpoints": volume})  # 8 hours - be generous!
def run_full_v1():
    """
    Full V1 experiment:
    1. Train baseline on 1K labeled
    2. Generate pseudo-labels for 6.5K unlabeled
    3. Train on 1K + 6.5K pseudo-labeled
    4. Compare to oracle (train on all 7.5K with real labels)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from collections import Counter
    import json

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    K = 8  # Updated from 4 based on pseudo-label check results (52% vs 38%)
    UNLABELED_SUBSET = 4000  # 4K is a good balance: 4x labeled, ~5-7 hours with batched gen

    print("=" * 60, flush=True)
    print(f"V1 FULL EXPERIMENT (K={K}, unlabeled={UNLABELED_SUBSET})", flush=True)
    print("=" * 60, flush=True)

    # Load data
    labeled, unlabeled_full, test = load_gsm8k_splits(labeled_size=1000)
    unlabeled = unlabeled_full.shuffle(seed=42).select(range(UNLABELED_SUBSET))

    # Helper to evaluate
    # 500 samples: SE ≈ 2.2%, 95% CI ≈ ±4.5% - good balance of speed and reliability
    def evaluate(model, tokenizer, test_data, name, sample_size=500):
        """Evaluate on test set."""
        model.eval()
        correct = 0
        sample = test_data.shuffle(seed=42).select(range(min(sample_size, len(test_data))))

        for example in sample:
            prompt = format_prompt(example["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = extract_answer(response)
            true = extract_answer(example["answer"])

            if answers_match(pred, true):
                correct += 1

        acc = correct / len(sample)
        print(f"{name}: {acc:.1%} ({correct}/{len(sample)})")
        return acc

    # Helper to train
    def train_sft(model, tokenizer, dataset, name, epochs=2):
        def format_example(example):
            prompt = format_prompt(example["question"])
            completion = example.get("answer") or example.get("pseudo_answer", "")
            return {"text": prompt + " " + completion}

        train_data = dataset.map(format_example)

        config = SFTConfig(
            output_dir=f"./{name}_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            logging_steps=100,
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=train_data,
            processing_class=tokenizer,
        )

        print(f"\nTraining {name} on {len(dataset)} examples...")
        trainer.train()
        return model

    # ---- Step 0: Base model (untrained) ----
    print("\n" + "=" * 40, flush=True)
    print("STEP 0: Base model (untrained, zero-shot)", flush=True)
    print("=" * 40, flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_acc = evaluate(model, tokenizer, test, "Base (untrained)")

    # ---- Step 1: Baseline (1K labeled) ----
    print("\n" + "=" * 40, flush=True)
    print("STEP 1: Baseline (train on 1K labeled)", flush=True)
    print("=" * 40, flush=True)

    model = train_sft(model, tokenizer, labeled, "baseline")
    baseline_acc = evaluate(model, tokenizer, test, "Baseline (1K SFT)")

    # ---- Step 2: Generate pseudo-labels ----
    print("\n" + "=" * 40)
    print("STEP 2: Pseudo-label 6.5K unlabeled")
    print("=" * 40)

    pseudo_labeled_data = []
    correct_pseudo = 0

    for i, example in enumerate(unlabeled):
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate K responses in ONE call (batched - much faster!)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=K,  # Generate all K at once
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        answers = [extract_answer(r) for r in responses]

        # Majority vote
        valid = [a for a in answers if a is not None]
        if valid:
            consensus = Counter(valid).most_common(1)[0][0]
            # Find a response that gave this answer
            for resp, ans in zip(responses, answers):
                if answers_match(ans, consensus):
                    # Create pseudo-labeled example
                    pseudo_labeled_data.append({
                        "question": example["question"],
                        "pseudo_answer": resp[len(prompt):].strip(),  # Just the completion
                        "answer": resp[len(prompt):].strip(),  # For compatibility
                    })
                    # Check accuracy (we secretly have ground truth)
                    true_ans = extract_answer(example["answer"])
                    if answers_match(consensus, true_ans):
                        correct_pseudo += 1
                    break

        if (i + 1) % 500 == 0:
            print(f"Pseudo-labeled {i + 1}/{len(unlabeled)}")

    pseudo_acc = correct_pseudo / len(pseudo_labeled_data) if pseudo_labeled_data else 0
    print(f"\nPseudo-labeled {len(pseudo_labeled_data)} examples")
    print(f"Pseudo-label accuracy: {pseudo_acc:.1%}")

    # ---- Step 3: Train on labeled + pseudo-labeled ----
    print("\n" + "=" * 40)
    print("STEP 3: Noisy Student (1K + 6.5K pseudo)")
    print("=" * 40)

    # Reload fresh model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    # Combine datasets
    combined = list(labeled) + pseudo_labeled_data
    combined_dataset = Dataset.from_list(combined)
    print(f"Combined dataset: {len(combined_dataset)} examples")

    model = train_sft(model, tokenizer, combined_dataset, "noisy_student")
    noisy_acc = evaluate(model, tokenizer, test, "Noisy Student")

    # ---- Step 3b: Pseudo-only (6.5K pseudo, no labeled) ----
    print("\n" + "=" * 40, flush=True)
    print("STEP 3b: Pseudo-only (6.5K pseudo, no labeled)", flush=True)
    print("=" * 40, flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    pseudo_only_dataset = Dataset.from_list(pseudo_labeled_data)
    print(f"Pseudo-only dataset: {len(pseudo_only_dataset)} examples", flush=True)

    model = train_sft(model, tokenizer, pseudo_only_dataset, "pseudo_only")
    pseudo_only_acc = evaluate(model, tokenizer, test, "Pseudo-only")

    # ---- Step 4: Oracle (all 7.5K with real labels) ----
    print("\n" + "=" * 40, flush=True)
    print(f"STEP 4: Oracle (1K + {UNLABELED_SUBSET} real labels)", flush=True)
    print("=" * 40, flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    # Combine labeled + unlabeled with real labels (same subset as noisy student for fair comparison)
    all_labeled = list(labeled) + list(unlabeled)
    all_dataset = Dataset.from_list(all_labeled)
    print(f"Oracle dataset: {len(all_dataset)} examples (1K + {UNLABELED_SUBSET} with real labels)", flush=True)

    model = train_sft(model, tokenizer, all_dataset, "oracle")
    oracle_acc = evaluate(model, tokenizer, test, "Oracle")

    # ---- Results ----
    print("\n" + "=" * 60, flush=True)
    print("V1 RESULTS (n=500 per eval, 95% CI ≈ ±4.5%)", flush=True)
    print("=" * 60, flush=True)
    print(f"Base (untrained):           {base_acc:.1%}", flush=True)
    print(f"Baseline (1K labeled):      {baseline_acc:.1%}", flush=True)
    print(f"Pseudo-only (6.5K pseudo):  {pseudo_only_acc:.1%}", flush=True)
    print(f"Noisy Student (1K + 6.5K):  {noisy_acc:.1%}", flush=True)
    print(f"Oracle (7.5K labeled):      {oracle_acc:.1%}", flush=True)
    print(f"Pseudo-label accuracy:      {pseudo_acc:.1%}", flush=True)
    print("=" * 60, flush=True)

    print(f"\n1K SFT improvement over base: +{baseline_acc - base_acc:.1%}", flush=True)
    print(f"Pseudo-only improvement over base: +{pseudo_only_acc - base_acc:.1%}", flush=True)
    if noisy_acc > baseline_acc:
        improvement = noisy_acc - baseline_acc
        recovery = (noisy_acc - baseline_acc) / (oracle_acc - baseline_acc) if oracle_acc > baseline_acc else 0
        print(f"Noisy Student improvement over baseline: +{improvement:.1%}", flush=True)
        print(f"Recovery toward oracle: {recovery:.1%}", flush=True)
    else:
        print("No improvement over baseline.", flush=True)

    results = {
        "base_acc": base_acc,
        "baseline_acc": baseline_acc,
        "pseudo_only_acc": pseudo_only_acc,
        "noisy_student_acc": noisy_acc,
        "oracle_acc": oracle_acc,
        "pseudo_label_acc": pseudo_acc,
        "pseudo_labeled_count": len(pseudo_labeled_data),
    }

    # Save results
    with open("/checkpoints/v1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()

    return results


# ============================================================================
# Oracle Only (for resuming after timeout)
# ============================================================================

@app.function(gpu="L4", timeout=60 * 60 * 2)  # 2 hours
def run_oracle_only():
    """
    Run just the Oracle step (for resuming after timeout).
    Uses same data splits as full V1 for consistency.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    UNLABELED_SUBSET = 4000  # Same as full V1

    print("=" * 60, flush=True)
    print(f"ORACLE ONLY (1K + {UNLABELED_SUBSET} real labels)", flush=True)
    print("=" * 60, flush=True)

    # Load data (same splits as full V1)
    labeled, unlabeled_full, test = load_gsm8k_splits(labeled_size=1000)
    unlabeled = unlabeled_full.shuffle(seed=42).select(range(UNLABELED_SUBSET))

    print(f"Labeled: {len(labeled)}, Unlabeled subset: {len(unlabeled)}, Test: {len(test)}")

    # Load fresh model
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Combine labeled + unlabeled with real labels
    all_labeled = list(labeled) + list(unlabeled)
    all_dataset = Dataset.from_list(all_labeled)
    print(f"Oracle dataset: {len(all_dataset)} examples", flush=True)

    # Train
    def format_example(example):
        prompt = format_prompt(example["question"])
        return {"text": prompt + " " + example["answer"]}

    train_data = all_dataset.map(format_example)

    config = SFTConfig(
        output_dir="./oracle_output",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    print("\nTraining oracle...", flush=True)
    trainer.train()

    # Evaluate
    print("\nEvaluating on test set...", flush=True)
    model.eval()
    correct = 0
    sample = test.shuffle(seed=42).select(range(500))

    for i, example in enumerate(sample):
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = extract_answer(response)
        true = extract_answer(example["answer"])

        if answers_match(pred, true):
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/500 - Running: {correct}/{i+1} ({correct/(i+1):.1%})", flush=True)

    oracle_acc = correct / len(sample)

    print("\n" + "=" * 60, flush=True)
    print(f"ORACLE RESULT: {oracle_acc:.1%} ({correct}/500)", flush=True)
    print("=" * 60, flush=True)

    # Print full V1 comparison (hardcoded from previous run)
    print("\nFull V1 Comparison:", flush=True)
    print(f"  Base (untrained):      21.0%", flush=True)
    print(f"  Baseline (1K):         27.0%", flush=True)
    print(f"  Pseudo-only (4K):      23.6%", flush=True)
    print(f"  Noisy Student (1K+4K): 26.8%", flush=True)
    print(f"  Oracle (1K+4K real):   {oracle_acc:.1%} <-- NEW", flush=True)

    return {"oracle_acc": oracle_acc, "correct": correct, "total": 500}


# ============================================================================
# V1.5: Confidence Filtering
# ============================================================================

@app.function(gpu="L4", timeout=60 * 60 * 8)  # 8 hours - be generous!
def run_v1_5_confidence_filtering():
    """
    V1.5: Test if high-confidence pseudo-labels help.

    Hypothesis: Quality > Quantity. The ~8% of examples with >=75% agreement
    (which were 100% accurate in pre-flight) may help more than all 4K at 52%.

    Conditions:
    1. Baseline (from V1): 27.0%
    2. Noisy Student (from V1): 26.8% (all pseudo-labels)
    3. Filtered Noisy Student: 1K real + high-conf pseudo only (NEW)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from collections import Counter

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    K = 8
    UNLABELED_SUBSET = 4000
    CONFIDENCE_THRESHOLD = 0.75  # 6/8 or more must agree

    print("=" * 60, flush=True)
    print(f"V1.5: CONFIDENCE FILTERING (threshold={CONFIDENCE_THRESHOLD:.0%})", flush=True)
    print("=" * 60, flush=True)

    # Load data (same splits as V1)
    labeled, unlabeled_full, test = load_gsm8k_splits(labeled_size=1000)
    unlabeled = unlabeled_full.shuffle(seed=42).select(range(UNLABELED_SUBSET))

    print(f"Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}, Test: {len(test)}", flush=True)

    # Load model and train baseline first
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Helper to train
    def train_sft(model, tokenizer, dataset, name, epochs=2):
        def format_example(example):
            prompt = format_prompt(example["question"])
            completion = example.get("answer") or example.get("pseudo_answer", "")
            return {"text": prompt + " " + completion}

        train_data = dataset.map(format_example)

        config = SFTConfig(
            output_dir=f"./{name}_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            logging_steps=100,
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=train_data,
            processing_class=tokenizer,
        )

        print(f"\nTraining {name} on {len(dataset)} examples...", flush=True)
        trainer.train()
        return model

    # Helper to evaluate
    def evaluate(model, tokenizer, test_data, name, sample_size=500):
        model.eval()
        correct = 0
        sample = test_data.shuffle(seed=42).select(range(min(sample_size, len(test_data))))

        for i, example in enumerate(sample):
            prompt = format_prompt(example["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = extract_answer(response)
            true = extract_answer(example["answer"])

            if answers_match(pred, true):
                correct += 1

            if (i + 1) % 100 == 0:
                print(f"  Eval {i + 1}/{sample_size}: {correct}/{i+1} ({correct/(i+1):.1%})", flush=True)

        acc = correct / len(sample)
        print(f"{name}: {acc:.1%} ({correct}/{len(sample)})", flush=True)
        return acc

    # ---- Step 1: Train baseline on 1K labeled ----
    print("\n" + "=" * 40, flush=True)
    print("STEP 1: Train baseline (1K labeled)", flush=True)
    print("=" * 40, flush=True)

    model = train_sft(model, tokenizer, labeled, "baseline")
    baseline_acc = evaluate(model, tokenizer, test, "Baseline")

    # ---- Step 2: Generate pseudo-labels WITH confidence tracking ----
    print("\n" + "=" * 40, flush=True)
    print("STEP 2: Pseudo-label with confidence tracking", flush=True)
    print("=" * 40, flush=True)

    all_pseudo = []  # All pseudo-labeled examples
    high_conf_pseudo = []  # Only high-confidence ones
    correct_all = 0
    correct_high_conf = 0

    confidence_distribution = {0.125: 0, 0.25: 0, 0.375: 0, 0.5: 0, 0.625: 0, 0.75: 0, 0.875: 0, 1.0: 0}

    for i, example in enumerate(unlabeled):
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate K responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=K,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        answers = [extract_answer(r) for r in responses]

        # Majority vote with confidence
        valid = [a for a in answers if a is not None]
        if valid:
            counter = Counter(valid)
            consensus, count = counter.most_common(1)[0]
            confidence = count / K

            # Track confidence distribution
            conf_bucket = round(confidence * 8) / 8  # Round to nearest 0.125
            if conf_bucket in confidence_distribution:
                confidence_distribution[conf_bucket] += 1

            # Find a response that gave this answer
            for resp, ans in zip(responses, answers):
                if answers_match(ans, consensus):
                    pseudo_example = {
                        "question": example["question"],
                        "answer": resp[len(prompt):].strip(),
                        "confidence": confidence,
                    }

                    # Check accuracy (we secretly have ground truth)
                    true_ans = extract_answer(example["answer"])
                    is_correct = answers_match(consensus, true_ans)

                    all_pseudo.append(pseudo_example)
                    if is_correct:
                        correct_all += 1

                    # Filter by confidence
                    if confidence >= CONFIDENCE_THRESHOLD:
                        high_conf_pseudo.append(pseudo_example)
                        if is_correct:
                            correct_high_conf += 1
                    break

        if (i + 1) % 500 == 0:
            print(f"Pseudo-labeled {i + 1}/{len(unlabeled)} - High-conf so far: {len(high_conf_pseudo)}", flush=True)

    # Report pseudo-labeling stats
    all_acc = correct_all / len(all_pseudo) if all_pseudo else 0
    high_conf_acc = correct_high_conf / len(high_conf_pseudo) if high_conf_pseudo else 0

    print(f"\nPseudo-labeling complete:", flush=True)
    print(f"  All pseudo-labels: {len(all_pseudo)} ({all_acc:.1%} accurate)", flush=True)
    print(f"  High-conf (>={CONFIDENCE_THRESHOLD:.0%}): {len(high_conf_pseudo)} ({high_conf_acc:.1%} accurate)", flush=True)
    print(f"\nConfidence distribution:", flush=True)
    for conf, count in sorted(confidence_distribution.items()):
        pct = count / len(all_pseudo) * 100 if all_pseudo else 0
        bar = "#" * int(pct / 2)
        print(f"  {conf:.0%}: {count:4d} ({pct:5.1f}%) {bar}", flush=True)

    # ---- Step 3: Train on 1K real + high-conf pseudo ----
    print("\n" + "=" * 40, flush=True)
    print(f"STEP 3: Train filtered model (1K + {len(high_conf_pseudo)} high-conf)", flush=True)
    print("=" * 40, flush=True)

    # Reload fresh model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    combined = list(labeled) + high_conf_pseudo
    combined_dataset = Dataset.from_list(combined)
    print(f"Filtered dataset: {len(combined_dataset)} examples (1K real + {len(high_conf_pseudo)} high-conf pseudo)", flush=True)

    model = train_sft(model, tokenizer, combined_dataset, "filtered_noisy_student")
    filtered_acc = evaluate(model, tokenizer, test, "Filtered Noisy Student")

    # ---- Results ----
    print("\n" + "=" * 60, flush=True)
    print("V1.5 RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"\nPseudo-label stats:", flush=True)
    print(f"  All: {len(all_pseudo)} examples, {all_acc:.1%} accurate", flush=True)
    print(f"  High-conf (>={CONFIDENCE_THRESHOLD:.0%}): {len(high_conf_pseudo)} examples, {high_conf_acc:.1%} accurate", flush=True)
    print(f"\nAccuracy comparison (n=500, 95% CI ~4.5%):", flush=True)
    print(f"  Base (untrained):           21.0% (from V1)", flush=True)
    print(f"  Baseline (1K):              {baseline_acc:.1%}", flush=True)
    print(f"  Noisy Student (1K+4K):      26.8% (from V1, all pseudo)", flush=True)
    print(f"  Filtered (1K+{len(high_conf_pseudo)} high-conf): {filtered_acc:.1%} <-- NEW", flush=True)
    print(f"  Oracle (1K+4K real):        32.4% (from V1)", flush=True)

    improvement = filtered_acc - baseline_acc
    if improvement > 0:
        print(f"\nFiltered Noisy Student improved over Baseline by +{improvement:.1%}!", flush=True)
        print("Quality > Quantity hypothesis SUPPORTED", flush=True)
    else:
        print(f"\nFiltered Noisy Student did not beat Baseline ({improvement:+.1%})", flush=True)
        if len(high_conf_pseudo) < 200:
            print(f"Only {len(high_conf_pseudo)} high-conf examples - may be too few", flush=True)

    return {
        "baseline_acc": baseline_acc,
        "filtered_acc": filtered_acc,
        "all_pseudo_count": len(all_pseudo),
        "all_pseudo_acc": all_acc,
        "high_conf_count": len(high_conf_pseudo),
        "high_conf_acc": high_conf_acc,
        "confidence_distribution": confidence_distribution,
    }


# ============================================================================
# Quick Smoke Test
# ============================================================================

@app.function(gpu="L4", timeout=60 * 15)  # 15 min
def run_smoke_test():
    """
    Quick test to verify batched generation and code changes work.
    Uses tiny amounts of data - should complete in ~5-10 minutes.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from collections import Counter

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    K = 8

    print("=" * 60, flush=True)
    print(f"SMOKE TEST (K={K})", flush=True)
    print("=" * 60, flush=True)

    # Load tiny subset
    labeled, unlabeled_full, test = load_gsm8k_splits(labeled_size=50)
    unlabeled = unlabeled_full.shuffle(seed=42).select(range(20))

    # Load model
    print("\nLoading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Test 1: Batched generation
    # Use same params as real experiment (256 tokens, not 128!)
    print("\n--- Test 1: Batched generation ---", flush=True)
    example = unlabeled[0]
    prompt = format_prompt(example["question"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Same as real experiment
            temperature=0.7,
            do_sample=True,
            num_return_sequences=K,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    answers = [extract_answer(r) for r in responses]
    print(f"Generated {len(responses)} responses", flush=True)
    print(f"Answers: {answers}", flush=True)

    valid = [a for a in answers if a is not None]
    if valid:
        consensus = Counter(valid).most_common(1)[0][0]
        print(f"Consensus: {consensus}", flush=True)

    # Test 2: Quick pseudo-labeling loop
    print("\n--- Test 2: Pseudo-labeling loop (5 examples) ---", flush=True)
    for i in range(5):
        example = unlabeled[i]
        prompt = format_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Same as real experiment
                temperature=0.7,
                do_sample=True,
                num_return_sequences=K,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        answers = [extract_answer(r) for r in responses]
        valid = [a for a in answers if a is not None]
        consensus = Counter(valid).most_common(1)[0][0] if valid else None

        true_ans = extract_answer(example["answer"])
        match = answers_match(consensus, true_ans)
        print(f"  [{i+1}] Consensus: {consensus}, True: {true_ans}, Match: {match}", flush=True)

    # Test 3: Quick SFT
    print("\n--- Test 3: Quick SFT (10 steps) ---", flush=True)

    def format_example(ex):
        return {"text": format_prompt(ex["question"]) + " " + ex["answer"]}

    train_data = labeled.select(range(10)).map(format_example)

    config = SFTConfig(
        output_dir="./smoke_test_output",
        max_steps=10,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    trainer.train()
    print("SFT completed!", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SMOKE TEST PASSED!", flush=True)
    print("=" * 60, flush=True)

    return {"status": "passed"}


# ============================================================================
# Entry point
# ============================================================================

@app.local_entrypoint()
def main():
    print("V1 Experiment Commands:")
    print("  modal run experiments/v1_simple.py::run_smoke_test")
    print("  modal run experiments/v1_simple.py::run_baseline")
    print("  modal run experiments/v1_simple.py::run_pseudo_label_check")
    print("  modal run experiments/v1_simple.py::run_full_v1")
