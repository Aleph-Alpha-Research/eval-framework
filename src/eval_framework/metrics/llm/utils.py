"""Utility functions for LLM-based metrics."""


def order_answers_for_comparison(candidate: str, reference: str, swap: bool) -> tuple[str, str]:
    """Order candidate and reference answers for A/B comparison.

    This function is used to mitigate position bias in LLM-as-judge evaluations
    by optionally swapping the order in which answers are presented.

    Args:
        candidate: The candidate completion to evaluate.
        reference: The reference/baseline completion.
        swap: If True, swap the order (reference becomes A, candidate becomes B).

    Returns:
        Tuple of (answer_a, answer_b) in the correct order.
    """
    if swap:
        return reference, candidate
    return candidate, reference
