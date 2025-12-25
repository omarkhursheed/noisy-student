"""
GRPO Debug - Run A: Structural Fix
Run with: modal run experiments/debug_grpo.py

Changes from previous runs:
- num_generations: 4 -> 8 (larger group for valid std)
- loss_type: dapo -> dr_grpo (fixes zero loss)
- max_completion_length: 1024 -> 512 (shrink box)
- learning_rate: 1e-6 -> 1e-5 (see trend in 30 steps)
- repetition_penalty: 1.2 (prevent degenerate loops)
- max_steps: 30 (observation run)

Success metrics to watch:
1. reward_std > 0
2. kl increasing
3. mean_terminated_length > 0
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.28",
    "trl>=0.8",
)

app = modal.App("debug-grpo", image=image)


@app.function(gpu="L4", timeout=60 * 30)
def debug_grpo():
    import torch
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import re

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    print("=" * 60)
    print("GRPO DEBUG - RUN A2: FOCUSED GENERATION")
    print("=" * 60)
    print("Config: temp=0.3, stop_strings=['####'], max_tokens=256, num_gen=8, dr_grpo")

    # Load larger dataset for more coverage
    ds = load_dataset("openai/gsm8k", "main")
    examples = ds["train"].select(range(100))

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompts
    def format_prompt(q):
        return f"Solve this math problem step by step. End with #### followed by the numerical answer.\n\nProblem: {q}\n\nSolution:"

    prompts = [format_prompt(ex["question"]) for ex in examples]
    train_dataset = Dataset.from_dict({"prompt": prompts})

    # Build answer lookup
    def extract_answer(text):
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '')
        numbers = re.findall(r'-?[\d,]+\.?\d*', text)
        return numbers[-1].replace(',', '') if numbers else None

    answer_lookup = {}
    for ex in examples:
        q = ex["question"].strip()
        # Also try with Problem: prefix removed
        answer_lookup[q] = extract_answer(ex["answer"])

    print(f"Answer lookup: {len(answer_lookup)} questions")

    # Debug: print first few entries
    for i, (q, a) in enumerate(list(answer_lookup.items())[:3]):
        print(f"  [{i}] Q: {q[:50]}... -> A: {a}")

    # Reward function with detailed logging
    call_count = [0]
    def reward_fn(completions, prompts, **kwargs):
        call_count[0] += 1
        rewards = []

        print(f"\n--- Reward call #{call_count[0]} ---")
        print(f"Batch size: {len(completions)}")

        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            # Extract question
            if "Problem:" in prompt and "Solution:" in prompt:
                start = prompt.index("Problem:") + len("Problem:")
                end = prompt.index("\n\nSolution:")
                question = prompt[start:end].strip()
            else:
                question = prompt.strip()

            target = answer_lookup.get(question)
            extracted = extract_answer(completion)

            match = extracted is not None and target is not None and abs(float(extracted) - float(target)) < 1e-6 if extracted and target else False
            reward = 1.0 if match else 0.0
            rewards.append(reward)

            # Log FULL completions for first 10 calls (to see everything)
            if call_count[0] <= 10:
                # Check for degenerate patterns
                has_repeat = "####" in completion and completion.count("####") > 2
                has_eos = tokenizer.eos_token in completion if hasattr(tokenizer, 'eos_token') else False

                print(f"\n  [{i}] len={len(completion)} chars, extracted={extracted}, target={target}, reward={reward}")
                print(f"      has_eos={has_eos}, repeats={'YES' if has_repeat else 'no'}")
                print(f"      --- FULL COMPLETION ---")
                print(completion)
                print(f"      --- END COMPLETION ---\n")

        print(f"Rewards: {rewards}")
        print(f"Mean: {sum(rewards)/len(rewards):.2f}, Variance: {sum((r - sum(rewards)/len(rewards))**2 for r in rewards)/len(rewards):.4f}")

        return rewards

    # Test 1: Check GRPOConfig defaults
    print("\n" + "=" * 60)
    print("TEST 1: GRPOConfig inspection")
    print("=" * 60)

    # Run A2: Focused generation
    config = GRPOConfig(
        output_dir="./debug_output",
        temperature=0.3,              # Low enough to stay coherent, high enough for diversity
        max_completion_length=256,    # GSM8K doesn't need more
        num_generations=8,
        loss_type="dr_grpo",
        learning_rate=1e-5,
        beta=0.04,
        per_device_train_batch_size=8,
        max_steps=50,
        logging_steps=1,
        report_to="none",
    )

    # Print all config values
    print("\nGRPOConfig values:")
    for key in ['num_generations', 'learning_rate', 'per_device_train_batch_size',
                'max_completion_length', 'temperature', 'beta', 'loss_type',
                'num_iterations', 'epsilon', 'max_grad_norm']:
        val = getattr(config, key, 'NOT_FOUND')
        print(f"  {key}: {val}")

    # Test 2: Run 30 steps (watch success metrics)
    print("\n" + "=" * 60)
    print("TEST 2: Run 30 training steps")
    print("Watch: reward_std > 0, kl increasing, mean_terminated_length > 0")
    print("=" * 60)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
    )

    # Check trainer internals
    print(f"\nTrainer info:")
    print(f"  Model device: {next(model.parameters()).device}")

    trainer.train()

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

    return {"status": "done", "reward_calls": call_count[0]}


@app.local_entrypoint()
def main():
    result = debug_grpo.remote()
    print(f"Result: {result}")
