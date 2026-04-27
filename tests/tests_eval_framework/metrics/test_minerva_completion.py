from eval_framework.metrics.completion.math_minerva_completion import MathMinervaCompletion
from eval_framework.metrics.completion.minerva_math_utils import extract_answers
from eval_framework.shared.types import Completion


def make_completion(completion: str, ground_truth: str | list[str]) -> Completion:
    """Helper to create Completion objects for testing."""
    return Completion(
        id=1,
        subject="test",
        ground_truth=ground_truth,
        prompt="test prompt",
        prompt_sequence_positions=None,
        messages=None,
        completion=completion,
        raw_completion=completion,
        raw_completion_sequence_positions=None,
    )


class TestMathMinervaCompletion:
    """Tests for Minerva final-answer extraction."""

    def test_extract_answers_defaults_to_english(self) -> None:
        english = "Final Answer: The final answer is 24. I hope it is correct."
        german = "Finale Antwort: Die finale Antwort lautet 24. Ich hoffe, die Antwort ist korrekt."

        assert extract_answers(english)[0] == "24"
        assert extract_answers(german)[0] != "24"

    def test_extract_answers_can_select_german(self) -> None:
        german = "Finale Antwort: Die finale Antwort lautet 24. Ich hoffe, die Antwort ist korrekt."

        assert extract_answers(german, cot_style="minerva_de")[0] == "24"

    def test_extract_answers_german_relaxed_selection(self) -> None:
        german = "Endgültige Antwort: die Antwort ist 24 ich hoffe, sie ist korrekt."

        assert extract_answers(german, cot_style="minerva_de", relaxed=True)[0] == "24"
        assert extract_answers(german, relaxed=True)[0] != "24"

    def test_metric_defaults_to_english_but_can_select_german(self) -> None:
        german = "Finale Antwort: Die finale Antwort lautet 24. Ich hoffe, die Antwort ist korrekt."
        default_results = MathMinervaCompletion().calculate(make_completion(german, "24"))

        german_metric = MathMinervaCompletion(cot_style="minerva_de")
        german_results = german_metric.calculate(make_completion(german, "24"))

        assert [result.value for result in default_results] == [0.0, 0.0]
        assert [result.value for result in german_results] == [1.0, 1.0]
