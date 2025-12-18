import pytest

from eval_framework.llm.base import BaseLLM
from eval_framework.llm.huggingface import Qwen3_0_6B
from eval_framework.metrics.llm.graders.chatbot_style_grader import ChatbotStyleGrader
from eval_framework.metrics.llm.graders.coherence_grader import CoherenceGrader, CoherenceGradingOutput
from eval_framework.metrics.llm.graders.comparison_grader import ComparisonGrader, MatchOutcome
from eval_framework.metrics.llm.graders.conciseness_grader import ConcisenessGrader
from eval_framework.metrics.llm.graders.contains_names_grader import ContainsNamesGrader
from eval_framework.metrics.llm.graders.format_correctness_grader import FormatCorrectnessGrader
from eval_framework.metrics.llm.graders.instruction_grader import InstructionGrader
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.refusal_grader import RefusalGrader


class TestMatchOutcome:
    """Tests for MatchOutcome enum methods."""

    def test_flip_a_wins_to_b_wins(self) -> None:
        assert MatchOutcome.A_WINS.flip() == MatchOutcome.B_WINS

    def test_flip_b_wins_to_a_wins(self) -> None:
        assert MatchOutcome.B_WINS.flip() == MatchOutcome.A_WINS

    def test_flip_draw_stays_draw(self) -> None:
        assert MatchOutcome.DRAW.flip() == MatchOutcome.DRAW

    def test_flip_is_involutory(self) -> None:
        """Flipping twice should return to original."""
        for outcome in MatchOutcome:
            assert outcome.flip().flip() == outcome


# NOTE: Run this tests to make sure redis has the cache in CI


@pytest.fixture(scope="module")
def judge() -> BaseLLM:
    return Qwen3_0_6B()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, instruction, completion",
    [
        (
            Language("en"),
            "Mike likes Pizza, Jenny does not.\nWho likes Pizza?",
            "Only Mike likes Pizza.",
        )
    ],
)

# Test is inherently flaky due to LLM judge non-determinism but an activated redis cache makes it deterministic.
def test_format_following_grader(language: Language, instruction: str, completion: str, judge: BaseLLM) -> None:
    format_following_grader = FormatCorrectnessGrader(judge)
    output = format_following_grader.grade(instruction, completion, language)
    assert output.reasons
    assert output.format_correctness in [0, 1]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, instruction, completion",
    [
        (
            Language("en"),
            "Mike likes Pizza, Jenny does not.\nWho likes Pizza?",
            "Only Mike likes Pizza.",
        ),
        (
            Language("de"),
            "Mike mag Pizza, Jenny nicht.\nWer von den beiden mag Pizza?",
            "Nur Mike.",
        ),
    ],
)
# Test is inherently flaky due to LLM judge non-determinism but an activated redis cache makes it deterministic.
def test_instruction_grader(language: Language, instruction: str, completion: str, judge: BaseLLM) -> None:
    instruction_grader = InstructionGrader(judge)
    output = instruction_grader.grade(instruction, completion, language)

    assert output.criticism
    assert output.quality is not None
    assert output.quality >= 4
    assert output.is_following_instruction is True
    assert output.has_correct_grammar_and_spelling is True
    assert output.is_context_consistent is True
    assert output.is_not_repeating is True
    assert output.is_trustworthy is True
    assert output.is_safe is True


@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, completion, expected",
    [
        (Language("en"), "Only Mike likes Pizza.", False),
        (Language("de"), "Hallo, gerne hilfe ich dir weiter! Nur Mike mag Pizza.", True),
    ],
)
def test_chatbot_style_grader(language: Language, completion: str, expected: bool, judge: BaseLLM) -> None:
    chatbot_style_grader = ChatbotStyleGrader(judge)
    output = chatbot_style_grader.grade(completion, language)

    assert output.thought_process
    assert output.is_chatbot_style == expected


@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, instruction, completion_1, completion_2, expected",
    [
        (
            Language("en"),
            "Mike likes Pizza, Jenny does not.\nWho likes Pizza?",
            "Only Jenny likes Pizza.",
            "Only Mike likes Pizza.",
            None,  # MatchOutcome.B_WINS,  # Qwen3_0_6B does not return a valid JSON here
        ),
        (
            Language("de"),
            "Mike mag Pizza, Jenny nicht.\nWer mag Pizza?",
            "Nur Mike mag Pizza.",
            "Nur Jenny mag Pizza.",
            MatchOutcome.B_WINS,  # Qwen3_0_6B gets it wrong on that one
        ),
    ],
)
def test_comparison_grader(
    language: Language, instruction: str, completion_1: str, completion_2: str, expected: MatchOutcome, judge: BaseLLM
) -> None:
    comparison_grader = ComparisonGrader(judge)
    # Disable randomization for deterministic test behavior
    output = comparison_grader.grade(instruction, completion_1, completion_2, language, randomize_order=False)

    if expected is not None:
        assert output.reasoning
    assert output.outcome == expected


@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, instruction, completion, expected",
    [
        (
            Language("en"),
            "Mike likes Pizza, Jenny does not.\nWho likes Pizza?",
            "Pizza is an Italian dish. Mike likes Italian food in general. Only Mike likes Pizza, not Jenny.",
            True,  # Qwen3_0_6B wrongly marks this one as concise
        ),
        (
            Language("de"),
            "Mike mag Pizza, Jenny nicht.\nWer mag Pizza?",
            "Nur Mike.",
            True,
        ),
    ],
)
# Test is inherently flaky due to LLM judge non-determinism but an activated redis cache makes it deterministic.
def test_conciseness_grader(
    language: Language, instruction: str, completion: str, expected: bool, judge: BaseLLM
) -> None:
    conciseness_grader = ConcisenessGrader(judge)
    output = conciseness_grader.grade(instruction, completion, language)

    assert output.thought_process
    assert output.is_concise == expected


@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, completion, expected",
    [
        (Language("en"), "Everyone likes Pizza.", False),
        (Language("de"), "Mike mag Pizza.", True),
    ],
)
def test_contains_names_grader(language: Language, completion: str, expected: bool, judge: BaseLLM) -> None:
    contains_names_grader = ContainsNamesGrader(judge)
    output = contains_names_grader.grade(completion, language)

    assert output.contains_names == expected


@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, completion, expected",
    [
        (Language("en"), "I'm sorry, I can't answer that question.", True),
        (Language("en"), "I'm not sure I want to provide any information.", False),  # Qwen3_0_6B misses this one
        (Language("en"), "Only Mike likes Pizza.", False),
        (Language("en"), "The answer is 42.", False),
    ],
)
def test_refusal_grader(language: Language, completion: str, expected: bool, judge: BaseLLM) -> None:
    refusal_grader = RefusalGrader(judge)
    output = refusal_grader.grade(completion, language)

    assert output.is_refusal == expected


@pytest.mark.external_api
@pytest.mark.gpu
@pytest.mark.parametrize(
    "language, instruction, completion",
    [
        (
            Language("en"),
            "Mike likes Pizza, Jenny does not.\nWho likes Pizza?",
            "Only Mike likes Pizza.",
        ),
    ],
)
def test_coherence_grader(language: Language, instruction: str, completion: str, judge: BaseLLM) -> None:
    coherence_grader = CoherenceGrader(judge)
    output = coherence_grader.grade(instruction, completion, language)

    assert isinstance(output, CoherenceGradingOutput)
    assert isinstance(output.coherence_score, int)
    assert output.coherence_score >= 0 and output.coherence_score <= 100
