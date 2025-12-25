"""
Simple GRPO training script for learning.
Run on Modal: modal run modal_app.py::train_grpo_simple
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ----- Configuration -----
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

# ----- Reward Function -----
def reward_fn(completions, prompts, **kwargs):
    """
    Compute rewards for a batch of completions.

    Args:
        completions: List of generated texts
        prompts: List of original prompts (to look up answers)

    Returns:
        List of reward floats
    """
    # Build prompt -> answer lookup
    answer_lookup = {d["prompt"]: d["answer"] for d in TRAIN_DATA}

    rewards = []
    for completion, prompt in zip(completions, prompts):
        correct_answer = answer_lookup.get(prompt, "")
        # Simple check: is the correct number in the response?
        if correct_answer and correct_answer in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

# ----- Main Training -----
def main():
    print(f"Loading model: {MODEL_NAME}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in TRAIN_DATA])

    # GRPO config
    config = GRPOConfig(
        output_dir="./grpo_output",
        num_generations=4,  # Group size
        learning_rate=1e-6,
        per_device_train_batch_size=2,
        max_steps=50,
        max_completion_length=64,
        temperature=0.7,
        logging_steps=10,
        report_to="none",  # Disable wandb for now
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    # Train!
    print("Starting GRPO training...")
    trainer.train()
    print("Training complete!")

    # Test the trained model
    print("\n----- Testing trained model -----")
    model.eval()
    for item in TRAIN_DATA[:3]:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=32, temperature=0.1, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {item['prompt']}")
        print(f"Response: {response}")
        print(f"Correct: {item['answer']}")
        print()

if __name__ == "__main__":
    main()
