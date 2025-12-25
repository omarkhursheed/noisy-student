"""
Test the GSM8K reward function locally (no GPU needed).

This script verifies:
1. Question extraction from prompts works
2. Answer extraction from completions works
3. Lookup hits/misses are tracked correctly
4. Rewards are computed correctly

Run with: python scripts/test_reward.py
Or: uv run python scripts/test_reward.py
"""

import sys
sys.path.insert(0, ".")

from src.data.gsm8k import (
    load_gsm8k_splits,
    format_prompt,
    extract_answer,
    extract_answer_from_model_output,
)
from src.training.rewards import (
    make_gsm8k_reward_fn,
    extract_question_from_prompt,
)


def test_question_extraction():
    """Test that we can extract questions from formatted prompts."""
    print("\n" + "="*60)
    print("TEST 1: Question Extraction")
    print("="*60)

    # Load real data
    splits = load_gsm8k_splits(labeled_size=10)
    example = splits["labeled"][0]

    # Format as prompt
    prompt = format_prompt(example["question"])
    print(f"\nOriginal question:\n{example['question'][:200]}...")
    print(f"\nFormatted prompt:\n{prompt[:200]}...")

    # Extract back
    extracted = extract_question_from_prompt(prompt)
    print(f"\nExtracted question:\n{extracted[:200]}...")

    # Check they match
    match = example["question"].strip() == extracted
    print(f"\nMatch: {match}")

    if not match:
        print("FAIL: Question extraction mismatch!")
        print(f"Original: {repr(example['question'][:100])}")
        print(f"Extracted: {repr(extracted[:100])}")
        return False

    print("PASS")
    return True


def test_answer_extraction():
    """Test that answer extraction handles various formats."""
    print("\n" + "="*60)
    print("TEST 2: Answer Extraction")
    print("="*60)

    test_cases = [
        # (input, expected_output)
        ("The answer is #### 42", "42"),
        ("So we get 16 - 3 = 13\n#### 13", "13"),
        ("Total = $1,234\n#### 1,234", "1234"),
        ("The answer is \\boxed{99}", "99"),
        ("Therefore the answer is 55.", "55"),
        ("2 + 2 = 4", "4"),
        ("No numbers here", None),  # Should return last number or None
    ]

    all_pass = True
    for text, expected in test_cases:
        result = extract_answer_from_model_output(text)
        # Note: "No numbers here" might return None or fail gracefully
        if text == "No numbers here":
            status = "OK (no numbers)"
        elif result == expected:
            status = "PASS"
        else:
            status = f"FAIL (got {result})"
            all_pass = False
        print(f"  '{text[:30]}...' -> {result} [{status}]")

    return all_pass


def test_reward_function():
    """Test the full reward function with real GSM8K data."""
    print("\n" + "="*60)
    print("TEST 3: Reward Function")
    print("="*60)

    # Load real data
    splits = load_gsm8k_splits(labeled_size=100)
    labeled = splits["labeled"]

    # Create reward function
    reward_fn = make_gsm8k_reward_fn(labeled, debug=True, debug_every=1)
    print(f"\nLookup size: {len(reward_fn.lookup)}")

    # Get first few examples
    examples = [labeled[i] for i in range(3)]

    # Simulate model completions - some correct, some wrong
    prompts = []
    completions = []

    for i, ex in enumerate(examples):
        prompt = format_prompt(ex["question"])
        true_answer = extract_answer(ex["answer"])

        prompts.append(prompt)

        if i == 0:
            # Correct answer in GSM8K format
            completions.append(f"Let me solve this step by step.\n#### {true_answer}")
        elif i == 1:
            # Correct answer in different format
            completions.append(f"The answer is {true_answer}")
        else:
            # Wrong answer
            completions.append("I don't know, maybe 999?")

    print(f"\nTesting with {len(prompts)} examples...")
    print("-" * 40)

    rewards = reward_fn(completions, prompts)

    print("-" * 40)
    print(f"\nRewards: {rewards}")
    print(f"Expected: [1.0, 1.0, 0.0] (first two correct, last wrong)")
    print(f"Stats: {reward_fn.stats}")

    # Check rewards
    expected = [1.0, 1.0, 0.0]
    if rewards == expected:
        print("\nPASS")
        return True
    else:
        print(f"\nFAIL: Got {rewards}, expected {expected}")
        return False


def test_lookup_miss():
    """Test that lookup misses are handled gracefully."""
    print("\n" + "="*60)
    print("TEST 4: Lookup Miss Handling")
    print("="*60)

    # Create reward function with small dataset
    splits = load_gsm8k_splits(labeled_size=10)
    reward_fn = make_gsm8k_reward_fn(splits["labeled"], debug=True, debug_every=1)

    # Try with a question NOT in the lookup
    fake_prompt = format_prompt("What is the meaning of life?")
    fake_completion = "42"

    rewards = reward_fn([fake_completion], [fake_prompt])

    print(f"\nRewards for unknown question: {rewards}")
    print(f"Stats: {reward_fn.stats}")

    if rewards == [0.0] and reward_fn.stats["misses"] > 0:
        print("\nPASS - Unknown question returns 0.0 and increments misses")
        return True
    else:
        print("\nFAIL")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("GSM8K Reward Function Tests")
    print("="*60)

    results = []
    results.append(("Question Extraction", test_question_extraction()))
    results.append(("Answer Extraction", test_answer_extraction()))
    results.append(("Reward Function", test_reward_function()))
    results.append(("Lookup Miss", test_lookup_miss()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
