"""
Reward functions for RLVR training.

This module provides reward functions that can be used with trl's GRPOTrainer.
The key insight: reward functions need access to ground truth, but the trainer
only passes (completions, prompts). We solve this with closures.

REWARD FUNCTION CONTRACT:
    Input:  completions (list[str]) - model-generated responses
            prompts (list[str]) - original prompts
            **kwargs - additional info from trainer
    Output: list[float] - rewards, same length as completions
"""

from typing import Callable

from ..data.gsm8k import extract_answer, extract_answer_from_model_output, answers_match


def extract_question_from_prompt(prompt: str) -> str:
    """
    Extract the original question from our formatted prompt.

    Our format: "Solve this math problem...Problem: {question}\n\nSolution:"

    Args:
        prompt: The full formatted prompt

    Returns:
        The extracted question text, or the full prompt if extraction fails
    """
    # Try to extract between "Problem:" and "Solution:"
    if "Problem:" in prompt and "Solution:" in prompt:
        try:
            start = prompt.index("Problem:") + len("Problem:")
            end = prompt.index("\n\nSolution:")
            return prompt[start:end].strip()
        except ValueError:
            pass

    # Fallback: return the whole prompt
    return prompt.strip()


def make_gsm8k_reward_fn(
    dataset,
    debug: bool = False,
    debug_every: int = 10,
) -> Callable:
    """
    Create a reward function for GSM8K with ground truth baked in.

    Args:
        dataset: HuggingFace dataset with 'question' and 'answer' fields
        debug: If True, print debug info during training
        debug_every: Print debug info every N calls (only if debug=True)

    Returns:
        Reward function compatible with GRPOTrainer

    Usage:
        reward_fn = make_gsm8k_reward_fn(train_dataset, debug=True)
        trainer = GRPOTrainer(..., reward_funcs=reward_fn)
    """
    # Build lookup: question -> answer
    # Using question text as key (not full prompt) for robustness
    lookup = {}
    for example in dataset:
        question = example["question"].strip()
        answer = extract_answer(example["answer"])
        if answer is not None:
            lookup[question] = answer

    print(f"Built reward lookup with {len(lookup)} questions")

    # Track stats for debugging
    call_count = [0]  # Use list to allow mutation in closure
    stats = {"hits": 0, "misses": 0, "correct": 0, "total": 0}

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Reward = 1.0 if extracted answer matches ground truth, else 0.0"""
        call_count[0] += 1
        rewards = []

        for completion, prompt in zip(completions, prompts):
            # Extract question from prompt
            question = extract_question_from_prompt(prompt)

            # Get ground truth
            true_answer = lookup.get(question)

            if true_answer is None:
                # Question not found in lookup - this is a bug
                stats["misses"] += 1
                if debug:
                    print(f"WARNING: Unknown question (first 50 chars): {question[:50]}...")
                rewards.append(0.0)
                continue

            stats["hits"] += 1
            stats["total"] += 1

            # Extract answer from model output
            extracted = extract_answer_from_model_output(completion)

            # Compare
            match = answers_match(extracted, true_answer)
            if match:
                stats["correct"] += 1

            rewards.append(1.0 if match else 0.0)

            # Debug logging
            if debug and call_count[0] % debug_every == 0:
                print(f"\n--- Reward Debug (call #{call_count[0]}) ---")
                print(f"Question: {question[:80]}...")
                print(f"Completion: {completion[:100]}...")
                print(f"Extracted: {extracted}, True: {true_answer}, Match: {match}")
                print(f"Stats: {stats['correct']}/{stats['total']} correct, {stats['misses']} misses")

        return rewards

    # Attach stats to function for later inspection
    reward_fn.stats = stats
    reward_fn.lookup = lookup

    return reward_fn


def make_pseudo_label_reward_fn(
    pseudo_labels: dict[str, str],
    debug: bool = False,
) -> Callable:
    """
    Create a reward function using pseudo-labels instead of ground truth.

    This is for noisy student training where we don't have real labels.

    Args:
        pseudo_labels: Dict mapping question -> pseudo-labeled answer
        debug: If True, print debug info

    Returns:
        Reward function compatible with GRPOTrainer
    """
    print(f"Built pseudo-label reward with {len(pseudo_labels)} questions")

    stats = {"hits": 0, "misses": 0, "correct": 0, "total": 0}

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Reward = 1.0 if extracted answer matches pseudo-label, else 0.0"""
        rewards = []

        for completion, prompt in zip(completions, prompts):
            question = extract_question_from_prompt(prompt)
            pseudo_answer = pseudo_labels.get(question)

            if pseudo_answer is None:
                stats["misses"] += 1
                rewards.append(0.0)
                continue

            stats["hits"] += 1
            stats["total"] += 1

            extracted = extract_answer_from_model_output(completion)
            match = answers_match(extracted, pseudo_answer)

            if match:
                stats["correct"] += 1

            rewards.append(1.0 if match else 0.0)

        return rewards

    reward_fn.stats = stats
    return reward_fn


# Quick test
if __name__ == "__main__":
    # Test question extraction
    print("Testing extract_question_from_prompt:")
    test_prompts = [
        "Solve this math problem step by step. End your answer with #### followed by the final numerical answer.\n\nProblem: What is 2 + 3?\n\nSolution:",
        "Problem: How many apples?\n\nSolution:",
        "Just a regular prompt without our format",
    ]

    for prompt in test_prompts:
        extracted = extract_question_from_prompt(prompt)
        print(f"  '{prompt[:40]}...' -> '{extracted}'")

    # Test with mock data
    print("\nTesting make_gsm8k_reward_fn:")

    class MockDataset:
        def __iter__(self):
            return iter([
                {"question": "What is 2 + 3?", "answer": "2 + 3 = 5\n#### 5"},
                {"question": "What is 10 - 4?", "answer": "10 - 4 = 6\n#### 6"},
            ])

    reward_fn = make_gsm8k_reward_fn(MockDataset(), debug=True, debug_every=1)

    # Test rewards
    completions = [
        "Let me solve this. 2 + 3 = 5. #### 5",  # Correct
        "The answer is 7",  # Wrong
    ]
    prompts = [
        "Problem: What is 2 + 3?\n\nSolution:",
        "Problem: What is 2 + 3?\n\nSolution:",
    ]

    rewards = reward_fn(completions, prompts)
    print(f"\nRewards: {rewards}")
    print(f"Expected: [1.0, 0.0]")
    print(f"Stats: {reward_fn.stats}")
