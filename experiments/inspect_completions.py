"""
Quick inspection of what completions look like at different temperatures.
Run with: modal run experiments/inspect_completions.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.28",
)

app = modal.App("inspect-completions", image=image)


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


@app.function(gpu="L4", timeout=60 * 10)
def inspect_completions():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import re

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load a few examples
    ds = load_dataset("openai/gsm8k", "main")

    for example_idx in [0, 5, 10]:
        example = ds["train"][example_idx]

        prompt = f"Solve this math problem step by step. End with #### followed by the numerical answer.\n\nProblem: {example['question']}\n\nSolution:"

        print(f"\n{'='*70}")
        print(f"EXAMPLE {example_idx}")
        print('='*70)
        print(f"\nQUESTION: {example['question'][:200]}...")

        # Extract ground truth answer
        gt_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', example['answer'])
        gt_answer = gt_match.group(1) if gt_match else "?"
        print(f"\nGROUND TRUTH ANSWER: {gt_answer}")
        print(f"GT answer length: {len(example['answer'])} chars")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs['input_ids'].shape[1]
        print(f"Prompt length: {prompt_len} tokens")

        # Test different settings
        settings = [
            (0.1, 256, "Eval (temp=0.1, 256 tok)"),
            (0.7, 256, "Explore (temp=0.7, 256 tok)"),
            (0.7, 512, "Explore (temp=0.7, 512 tok)"),
            (0.7, 768, "Explore (temp=0.7, 768 tok)"),
        ]

        for temp, max_tokens, label in settings:
            print(f"\n{'-'*50}")
            print(f"{label}")
            print('-'*50)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = response[len(prompt):]

            # Count tokens
            completion_tokens = len(tokenizer.encode(completion))
            hit_limit = completion_tokens >= max_tokens - 5

            # Check for #### answer - use same extraction as V1
            has_answer = "####" in completion
            extracted = extract_answer(completion)  # Use full extraction with fallbacks

            print(f"Length: {len(completion)} chars, {completion_tokens} tokens")
            print(f"Hit limit: {hit_limit}")
            print(f"Has ####: {has_answer}")
            print(f"Extracted answer: {extracted}")
            print(f"Correct: {extracted == gt_answer if extracted else False}")

            # Show completion
            print(f"\n--- COMPLETION ---")
            if len(completion) <= 800:
                print(completion)
            else:
                print(completion[:600])
                print(f"\n... [{len(completion) - 800} chars omitted] ...\n")
                print(completion[-200:])

    return "Done"


@app.local_entrypoint()
def main():
    inspect_completions.remote()
