"""
Modal configuration for GPU compute.

Modal is a serverless platform - you only pay for actual GPU time used.
No idle costs, no setup. Code runs in containers on their cloud.

COST REFERENCE (as of late 2024):
- T4 (16GB):    ~$0.59/hr  - Good for testing small things
- L4 (24GB):    ~$0.80/hr  - Best value for 1-3B models
- A10G (24GB):  ~$1.10/hr  - Faster than L4, good for training
- A100 (40GB):  ~$2.78/hr  - For 7B+ models
- H100 (80GB):  ~$4.25/hr  - Overkill for now

STRATEGY: Start with L4 for development, upgrade only when needed.
With $1530 in credits, that's ~1900 hours of L4 time.

HOW MODAL WORKS:
1. You define a container image (dependencies, etc.)
2. You decorate functions with @app.function() to run them on Modal
3. When you call the function, Modal spins up a container and runs it
4. You only pay for the time your function runs

To run: `modal run modal_app.py`
To deploy: `modal deploy modal_app.py`
"""

import modal

# ---------------------------------------------------------------------------
# STEP 1: Define our container image
# ---------------------------------------------------------------------------
# This is like a Dockerfile but in Python. Modal caches layers, so rebuilds
# are fast after the first time.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML stack
        "torch>=2.0",
        "transformers>=4.40",
        "datasets>=2.18",
        "accelerate>=0.28",
        # RL libraries
        "trl>=0.8",
        "peft>=0.10",
        # Utilities
        "wandb>=0.16",
        "pyyaml>=6.0",
    )
    # Add our local source code to the image
    .add_local_python_source("src")
)

# ---------------------------------------------------------------------------
# STEP 2: Create the Modal app
# ---------------------------------------------------------------------------

app = modal.App("noisy-student", image=image)

# ---------------------------------------------------------------------------
# STEP 3: Define GPU functions
# ---------------------------------------------------------------------------
# Each function can have different GPU requirements.
# Start cheap (L4), upgrade only when you hit memory limits or need speed.


@app.function(gpu="L4", timeout=60 * 60)  # 1 hour timeout
def test_gpu_setup():
    """
    Quick test to verify GPU access works.
    Run with: modal run modal_app.py::test_gpu_setup

    This costs about $0.01 to run (less than a minute of L4 time).
    """
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Quick transformers test
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B"  # Tiny model for testing
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to("cuda")
    print("Model loaded successfully!")

    # Quick generation test
    inputs = tokenizer("2 + 2 =", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10)
    print(f"Test generation: {tokenizer.decode(outputs[0])}")

    return "GPU setup verified!"


@app.function(gpu="L4", timeout=60 * 60)  # 1 hour timeout
def train_grpo_simple():
    """
    Simple GRPO training on arithmetic - for learning how trl works.
    Run with: modal run modal_app.py::train_grpo_simple

    Cost estimate: ~$0.10-0.20 (5-15 minutes of L4 time)
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    TRAIN_DATA = [
        {"prompt": "What is 2 + 3?", "answer": "5"},
        {"prompt": "What is 7 - 4?", "answer": "3"},
        {"prompt": "What is 5 * 2?", "answer": "10"},
        {"prompt": "What is 8 / 2?", "answer": "4"},
        {"prompt": "What is 10 + 5?", "answer": "15"},
        {"prompt": "What is 9 - 3?", "answer": "6"},
        {"prompt": "What is 4 * 3?", "answer": "12"},
        {"prompt": "What is 12 / 4?", "answer": "3"},
    ]

    def reward_fn(completions, prompts, **kwargs):
        """Reward = 1 if correct answer in response, else 0."""
        answer_lookup = {d["prompt"]: d["answer"] for d in TRAIN_DATA}
        rewards = []
        for completion, prompt in zip(completions, prompts):
            correct = answer_lookup.get(prompt, "")
            rewards.append(1.0 if correct and correct in completion else 0.0)
        return rewards

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in TRAIN_DATA])

    config = GRPOConfig(
        output_dir="./grpo_output",
        num_generations=4,
        learning_rate=1e-6,
        per_device_train_batch_size=4,  # Must be divisible by num_generations
        max_steps=50,
        max_completion_length=64,
        temperature=0.7,
        logging_steps=10,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    print("Starting GRPO training...")
    trainer.train()
    print("Training complete!")

    # Test
    print("\n----- Testing trained model -----")
    model.eval()
    for item in TRAIN_DATA[:3]:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=32, temperature=0.1, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Q: {item['prompt']}")
        print(f"A: {response}")
        print(f"Expected: {item['answer']}\n")

    return "GRPO training complete!"


@app.function(gpu="L4", timeout=60 * 60 * 4)  # 4 hour timeout
def train_rl(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    dataset_name: str = "gsm8k",
    num_train_steps: int = 100,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    use_lora: bool = True,
):
    """
    Run RL training on Modal.
    This is a placeholder - we'll fill in the actual training logic later.

    Run with: modal run modal_app.py::train_rl
    """
    # TODO: Implement actual training
    print(f"Would train {model_name} on {dataset_name}")
    print(f"Steps: {num_train_steps}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"LoRA: {use_lora}")
    return "Training placeholder - implement me!"


@app.function(gpu="L4", timeout=60 * 60 * 2)  # 2 hour timeout
def generate_pseudo_labels(
    model_path: str,
    dataset_name: str = "gsm8k",
    num_samples_per_prompt: int = 16,
    temperature: float = 0.7,
):
    """
    Generate multiple responses per prompt and compute consensus labels.
    This is a placeholder - we'll fill in the actual logic later.

    Run with: modal run modal_app.py::generate_pseudo_labels
    """
    # TODO: Implement pseudo-labeling
    print(f"Would generate {num_samples_per_prompt} samples per prompt")
    print(f"Model: {model_path}, Dataset: {dataset_name}")
    return "Pseudo-labeling placeholder - implement me!"


@app.function(gpu="L4", timeout=60 * 60 * 4)  # 4 hour timeout
def train_grpo_gsm8k(
    labeled_size: int = 100,
    max_steps: int = 50,
    learning_rate: float = 1e-6,
    num_generations: int = 4,
    debug: bool = True,
):
    """
    GRPO training on GSM8K with verifiable rewards.

    This is the core RLVR loop:
    1. Model generates completions for math problems
    2. We extract the answer and compare to ground truth
    3. Reward = 1 if correct, 0 if wrong
    4. GRPO updates model to make correct answers more likely

    Run with: modal run modal_app.py::train_grpo_gsm8k

    Args:
        labeled_size: Number of training examples to use
        max_steps: Training steps
        learning_rate: LR for optimizer
        num_generations: K completions per prompt for GRPO groups
        debug: Print debug info during training
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Import our modules (Modal adds src to Python path via add_local_python_source)
    from src.data.gsm8k import load_gsm8k_splits, format_prompt, extract_answer_from_model_output
    from src.training.rewards import make_gsm8k_reward_fn

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"=" * 60)
    print(f"GRPO Training on GSM8K")
    print(f"=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Labeled size: {labeled_size}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num generations (K): {num_generations}")
    print(f"=" * 60)

    # ----- Load Data -----
    print("\nLoading GSM8K...")
    splits = load_gsm8k_splits(labeled_size=labeled_size)
    train_data = splits["labeled"]
    test_data = splits["test"]

    # ----- Load Model -----
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ----- Create Dataset -----
    # GRPO expects a dataset with 'prompt' field
    def make_prompt_dataset(data):
        prompts = [format_prompt(ex["question"]) for ex in data]
        return Dataset.from_dict({"prompt": prompts})

    train_dataset = make_prompt_dataset(train_data)
    print(f"Train dataset size: {len(train_dataset)}")

    # ----- Create Reward Function -----
    reward_fn = make_gsm8k_reward_fn(train_data, debug=debug, debug_every=10)

    # ----- GRPO Config -----
    config = GRPOConfig(
        output_dir="./grpo_gsm8k_output",
        num_generations=num_generations,
        learning_rate=learning_rate,
        per_device_train_batch_size=num_generations,  # Must be divisible by num_generations
        max_steps=max_steps,
        max_completion_length=512,  # Math solutions need room for step-by-step reasoning
        temperature=0.7,
        logging_steps=10,
        report_to="none",  # Disable wandb for now
        # Gradient accumulation to simulate larger batches
        gradient_accumulation_steps=2,
    )

    # ----- Create Trainer -----
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
    )

    # ----- Train! -----
    print("\nStarting GRPO training...")
    trainer.train()
    print("Training complete!")

    # ----- Print Reward Stats -----
    print(f"\n{'=' * 60}")
    print("Reward Function Stats:")
    print(f"  Hits: {reward_fn.stats['hits']}")
    print(f"  Misses: {reward_fn.stats['misses']}")
    print(f"  Correct: {reward_fn.stats['correct']}/{reward_fn.stats['total']}")
    if reward_fn.stats['total'] > 0:
        acc = reward_fn.stats['correct'] / reward_fn.stats['total']
        print(f"  Accuracy: {acc:.1%}")

    # ----- Evaluate on Test Set -----
    print(f"\n{'=' * 60}")
    print("Evaluating on test set (first 50 examples)...")
    model.eval()

    eval_correct = 0
    eval_total = 50

    for i in range(eval_total):
        example = test_data[i]
        prompt = format_prompt(example["question"])
        true_answer = example["answer"].split("####")[-1].strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):]  # Remove prompt
        extracted = extract_answer_from_model_output(completion)

        try:
            if extracted and abs(float(extracted) - float(true_answer)) < 1e-6:
                eval_correct += 1
        except (ValueError, TypeError):
            pass

        if i < 3:  # Show first 3 examples
            print(f"\nExample {i+1}:")
            print(f"  Question: {example['question'][:80]}...")
            print(f"  Completion: {completion[:100]}...")
            print(f"  Extracted: {extracted}, True: {true_answer}")

    print(f"\n{'=' * 60}")
    print(f"Test Accuracy: {eval_correct}/{eval_total} = {eval_correct/eval_total:.1%}")
    print(f"{'=' * 60}")

    return {
        "train_accuracy": reward_fn.stats['correct'] / max(reward_fn.stats['total'], 1),
        "test_accuracy": eval_correct / eval_total,
        "reward_stats": reward_fn.stats,
    }


# ---------------------------------------------------------------------------
# STEP 4: Local entrypoint for testing
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """
    Default entrypoint when you run `modal run modal_app.py`
    Runs the GPU test to verify everything works.
    """
    print("Testing GPU setup on Modal...")
    result = test_gpu_setup.remote()
    print(f"\nResult: {result}")


# ---------------------------------------------------------------------------
# NOTES ON MODAL PATTERNS
# ---------------------------------------------------------------------------
#
# 1. REMOTE CALLS:
#    result = my_function.remote(arg1, arg2)  # Runs on Modal cloud
#    result = my_function.local(arg1, arg2)   # Runs locally (no GPU)
#
# 2. VOLUMES (for saving checkpoints):
#    volume = modal.Volume.from_name("my-volume", create_if_missing=True)
#    @app.function(volumes={"/data": volume})
#    def save_checkpoint(): ...
#
# 3. SECRETS (for wandb, HuggingFace):
#    modal secret create wandb WANDB_API_KEY=xxx
#    @app.function(secrets=[modal.Secret.from_name("wandb")])
#    def train(): ...
#
# 4. MULTIPLE GPUS:
#    @app.function(gpu="A100", num_gpus=2)
#    def distributed_train(): ...
