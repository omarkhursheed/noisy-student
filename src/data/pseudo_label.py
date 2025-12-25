"""
Pseudo-labeling via consensus voting.

Core idea:
1. Generate K responses per prompt
2. Extract answer from each response
3. Take majority vote as pseudo-label
4. Optionally filter by confidence (agreement ratio)
"""

from collections import Counter
from dataclasses import dataclass

from .gsm8k import extract_answer_from_model_output, answers_match


@dataclass
class PseudoLabel:
    """Result of pseudo-labeling a single example."""
    prompt: str
    consensus_answer: str | None
    confidence: float  # Fraction of responses agreeing with consensus
    num_responses: int
    all_answers: list[str | None]  # For debugging
    best_response: str | None  # A response that gave the consensus answer


def majority_vote(answers: list[str | None]) -> tuple[str | None, float]:
    """
    Find the most common answer and its frequency.

    Args:
        answers: List of extracted answers (may contain None)

    Returns:
        (consensus_answer, confidence) where confidence is fraction agreeing
    """
    # Filter out None answers
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None, 0.0

    # Count occurrences
    counter = Counter(valid_answers)
    most_common, count = counter.most_common(1)[0]

    confidence = count / len(answers)  # Use total answers, not just valid
    return most_common, confidence


def pseudo_label_single(
    prompt: str,
    responses: list[str],
) -> PseudoLabel:
    """
    Generate pseudo-label for a single prompt given multiple responses.

    Args:
        prompt: The original prompt
        responses: List of K model-generated responses

    Returns:
        PseudoLabel with consensus answer and confidence
    """
    # Extract answers from all responses
    answers = [extract_answer_from_model_output(r) for r in responses]

    # Get majority vote
    consensus, confidence = majority_vote(answers)

    # Find a response that gave the consensus answer (for SFT training)
    best_response = None
    if consensus is not None:
        for response, answer in zip(responses, answers):
            if answers_match(answer, consensus):
                best_response = response
                break

    return PseudoLabel(
        prompt=prompt,
        consensus_answer=consensus,
        confidence=confidence,
        num_responses=len(responses),
        all_answers=answers,
        best_response=best_response,
    )


def filter_by_confidence(
    pseudo_labels: list[PseudoLabel],
    min_confidence: float = 0.5,
) -> list[PseudoLabel]:
    """
    Filter pseudo-labels to only keep high-confidence ones.

    Args:
        pseudo_labels: List of PseudoLabel objects
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list
    """
    filtered = [pl for pl in pseudo_labels if pl.confidence >= min_confidence]
    print(f"Filtered {len(pseudo_labels)} -> {len(filtered)} pseudo-labels (min_confidence={min_confidence})")
    return filtered


def compute_pseudo_label_accuracy(
    pseudo_labels: list[PseudoLabel],
    ground_truth: list[str],
) -> dict:
    """
    Evaluate pseudo-label quality against ground truth.

    Args:
        pseudo_labels: List of PseudoLabel objects
        ground_truth: List of true answers

    Returns:
        Dict with accuracy metrics
    """
    assert len(pseudo_labels) == len(ground_truth)

    correct = 0
    total_with_consensus = 0

    for pl, true_answer in zip(pseudo_labels, ground_truth):
        if pl.consensus_answer is not None:
            total_with_consensus += 1
            if answers_match(pl.consensus_answer, true_answer):
                correct += 1

    accuracy = correct / total_with_consensus if total_with_consensus > 0 else 0
    coverage = total_with_consensus / len(pseudo_labels)

    return {
        "accuracy": accuracy,  # Of those with consensus, how many correct?
        "coverage": coverage,  # What fraction got a consensus answer?
        "correct": correct,
        "total_with_consensus": total_with_consensus,
        "total": len(pseudo_labels),
    }


# Quick test
if __name__ == "__main__":
    # Test majority vote
    print("Testing majority vote:")

    test_cases = [
        (["42", "42", "42", "43"], ("42", 0.75)),
        (["10", "20", "10", "30"], ("10", 0.5)),
        ([None, None, "5", "5"], ("5", 0.5)),
        ([None, None, None, None], (None, 0.0)),
    ]

    for answers, expected in test_cases:
        result = majority_vote(answers)
        status = "OK" if result == expected else f"FAIL (got {result})"
        print(f"  {answers} -> {result} {status}")

    # Test pseudo-labeling
    print("\nTesting pseudo-labeling:")
    responses = [
        "Let me solve this. 2 + 3 = 5. #### 5",
        "The answer is \\boxed{5}",
        "2 + 3 = 5",
        "I think it's 6",  # Wrong answer
    ]
    pl = pseudo_label_single("What is 2+3?", responses)
    print(f"  Consensus: {pl.consensus_answer}")
    print(f"  Confidence: {pl.confidence}")
    print(f"  All answers: {pl.all_answers}")
