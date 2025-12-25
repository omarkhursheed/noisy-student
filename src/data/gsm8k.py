"""
GSM8K dataset loading and answer extraction.

GSM8K format:
- question: The math word problem
- answer: Step-by-step solution ending with "#### <final_answer>"

Example answer:
"Janet's ducks lay 16 eggs per day. She eats 3 for breakfast...
So she has 16 - 3 - 4 = 9 eggs left.
9 * 2 = $18 per day.
#### 18"
"""

import re
from datasets import load_dataset


def extract_answer(solution: str) -> str | None:
    """
    Extract the final numerical answer from a GSM8K solution.

    GSM8K format: solution ends with "#### <number>"

    Args:
        solution: The full solution text

    Returns:
        The extracted answer as a string, or None if not found
    """
    # Look for #### followed by the answer
    # Pattern: optional minus, digits with commas, optional decimal part (requires digit after .)
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', solution)
    if match:
        # Remove commas from numbers like "1,000"
        return match.group(1).replace(',', '')
    return None


def extract_answer_from_model_output(text: str) -> str | None:
    """
    Extract answer from model-generated text.

    Models might output in different formats:
    - "#### 42"
    - "The answer is 42"
    - "\\boxed{42}"
    - Just "42" at the end

    Args:
        text: Model-generated response

    Returns:
        Extracted answer or None
    """
    # Number pattern: optional minus, digits with commas, optional decimal (requires digit after .)
    num_pattern = r'-?[\d,]+(?:\.\d+)?'

    # Try #### format first (GSM8K style)
    match = re.search(r'####\s*(' + num_pattern + r')', text)
    if match:
        return match.group(1).replace(',', '')

    # Try \boxed{} format (common in math models)
    match = re.search(r'\\boxed\{(' + num_pattern + r')\}', text)
    if match:
        return match.group(1).replace(',', '')

    # Try "answer is X" format
    match = re.search(r'answer is[:\s]*(' + num_pattern + r')', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # Try "= X" at end of text
    match = re.search(r'=\s*(' + num_pattern + r')\s*$', text)
    if match:
        return match.group(1).replace(',', '')

    # Last resort: find the last number in the text
    numbers = re.findall(num_pattern, text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def answers_match(answer1: str | None, answer2: str | None) -> bool:
    """
    Check if two answers match (handling float comparison).
    """
    if answer1 is None or answer2 is None:
        return False

    try:
        # Try numeric comparison
        return abs(float(answer1) - float(answer2)) < 1e-6
    except ValueError:
        # Fall back to string comparison
        return answer1.strip() == answer2.strip()


def load_gsm8k_splits(
    labeled_size: int = 1000,
    seed: int = 42,
):
    """
    Load GSM8K and create labeled/unlabeled splits.

    Args:
        labeled_size: Number of examples to use as "labeled"
        seed: Random seed for reproducibility

    Returns:
        dict with keys:
        - labeled: Dataset with labeled_size examples (use real labels)
        - unlabeled: Dataset with remaining examples (pretend no labels)
        - test: Full test set
    """
    # Load full dataset
    dataset = load_dataset("openai/gsm8k", "main")

    train_data = dataset["train"]
    test_data = dataset["test"]

    # Shuffle and split train
    train_data = train_data.shuffle(seed=seed)

    labeled = train_data.select(range(labeled_size))
    unlabeled = train_data.select(range(labeled_size, len(train_data)))

    print(f"GSM8K splits:")
    print(f"  Labeled:   {len(labeled)} examples")
    print(f"  Unlabeled: {len(unlabeled)} examples")
    print(f"  Test:      {len(test_data)} examples")

    return {
        "labeled": labeled,
        "unlabeled": unlabeled,
        "test": test_data,
    }


def format_prompt(question: str) -> str:
    """
    Format a GSM8K question as a prompt for the model.
    """
    return f"Solve this math problem step by step. End your answer with #### followed by the final numerical answer.\n\nProblem: {question}\n\nSolution:"


def format_for_sft(example: dict) -> dict:
    """
    Format a GSM8K example for SFT training.

    Args:
        example: Dict with 'question' and 'answer' keys

    Returns:
        Dict with 'prompt' and 'completion' keys
    """
    return {
        "prompt": format_prompt(example["question"]),
        "completion": example["answer"],
    }


# Quick test
if __name__ == "__main__":
    # Test answer extraction
    test_cases = [
        ("The answer is #### 42", "42"),
        ("So we get 16 - 3 = 13\n#### 13", "13"),
        ("Total = $1,234\n#### 1,234", "1234"),
        ("The answer is \\boxed{99}", "99"),
        ("Therefore the answer is 55.", "55"),
    ]

    print("Testing answer extraction:")
    for text, expected in test_cases:
        result = extract_answer_from_model_output(text)
        status = "OK" if result == expected else f"FAIL (got {result})"
        print(f"  '{text[:30]}...' -> {result} {status}")

    # Test loading
    print("\nTesting data loading:")
    splits = load_gsm8k_splits(labeled_size=100)

    # Show example
    example = splits["labeled"][0]
    print(f"\nExample question: {example['question'][:100]}...")
    print(f"Example answer: {example['answer'][:100]}...")
    extracted = extract_answer(example["answer"])
    print(f"Extracted answer: {extracted}")
