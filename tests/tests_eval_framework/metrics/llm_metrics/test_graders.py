import json
from collections.abc import Sequence

import pytest
from openai.types.chat.completion_create_params import ResponseFormat

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm.graders.chatbot_style_grader import ChatbotStyleGrader
from eval_framework.metrics.llm.graders.coherence_grader import CoherenceGrader
from eval_framework.metrics.llm.graders.comparison_grader import ComparisonGrader, MatchOutcome
from eval_framework.metrics.llm.graders.conciseness_grader import ConcisenessGrader
from eval_framework.metrics.llm.graders.contains_names_grader import ContainsNamesGrader
from eval_framework.metrics.llm.graders.format_correctness_grader import FormatCorrectnessGrader
from eval_framework.metrics.llm.graders.instruction_grader import InstructionGrader
from eval_framework.metrics.llm.graders.language import Language, LanguageNotSupportedError
from eval_framework.metrics.llm.graders.refusal_grader import RefusalGrader
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import Message, Role

INSTRUCTION = "Mike likes Pizza, Jenny does not.\nWho likes Pizza?"
COMPLETION = "Only Mike likes Pizza."


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


class ScriptedJudge(BaseLLM):
    """A judge that answers every prompt with a fixed response.

    Lets the grader tests cover prompt assembly and response parsing deterministically, without a model.
    """

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[Sequence[Message]] = []

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: ResponseFormat | None = None,
    ) -> list[RawCompletion]:
        self.prompts.extend(messages)
        return [
            RawCompletion(
                prompt="\n".join(message.content for message in single_messages),
                prompt_num_tokens=0,
                completion=self.response,
                completion_num_tokens=0,
            )
            for single_messages in messages
        ]

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        raise NotImplementedError

    def message(self, role: Role) -> str:
        """Content of the given role in the single prompt this judge received."""
        assert len(self.prompts) == 1
        return next(message.content for message in self.prompts[0] if message.role == role)


@pytest.mark.parametrize(
    "response, expected",
    [
        ('{"is_refusal": true}', True),
        ('{"is_refusal": false}', False),
        ("I cannot tell.", None),
    ],
)
def test_refusal_grader(response: str, expected: bool | None) -> None:
    judge = ScriptedJudge(response)
    output = RefusalGrader(judge).grade(COMPLETION, Language("en"))

    assert output.is_refusal == expected
    assert output.judge_response == response
    assert COMPLETION in judge.message(Role.USER)


def test_refusal_grader_rejects_unsupported_language() -> None:
    with pytest.raises(LanguageNotSupportedError):
        RefusalGrader(ScriptedJudge('{"is_refusal": true}')).grade(COMPLETION, Language("de"))


@pytest.mark.parametrize(
    "language, expected_prompt_start",
    [
        (Language("en"), "Your assignment is to assess whether text contains names"),
        (Language("de"), "Deine Aufgabe ist es zu bewerten, ob ein Text Namen"),
    ],
)
def test_contains_names_grader_uses_language_specific_prompt(language: Language, expected_prompt_start: str) -> None:
    judge = ScriptedJudge('{"contains_names": true}')
    output = ContainsNamesGrader(judge).grade("Mike mag Pizza.", language)

    assert output.contains_names is True
    assert judge.message(Role.SYSTEM).startswith(expected_prompt_start)
    assert "Mike mag Pizza." in judge.message(Role.USER)


@pytest.mark.parametrize("language", [Language("en"), Language("de")])
@pytest.mark.parametrize("is_chatbot_style", [True, False])
def test_chatbot_style_grader(language: Language, is_chatbot_style: bool) -> None:
    judge = ScriptedJudge(json.dumps({"thought_process": "greets the user", "is_chatbot_style": is_chatbot_style}))
    output = ChatbotStyleGrader(judge).grade(COMPLETION, language)

    assert output.thought_process == "greets the user"
    assert output.is_chatbot_style is is_chatbot_style


@pytest.mark.parametrize("language", [Language("en"), Language("de")])
@pytest.mark.parametrize("is_concise", [True, False])
def test_conciseness_grader(language: Language, is_concise: bool) -> None:
    judge = ScriptedJudge(json.dumps({"thought_process": "short answer", "is_concise": is_concise}))
    output = ConcisenessGrader(judge).grade(INSTRUCTION, COMPLETION, language)

    assert output.thought_process == "short answer"
    assert output.is_concise is is_concise
    assert INSTRUCTION in judge.message(Role.USER)
    assert COMPLETION in judge.message(Role.USER)


@pytest.mark.parametrize(
    "language, verdict, expected",
    [
        (Language("en"), "Answer 1 is better", MatchOutcome.A_WINS),
        (Language("en"), "Answer 2 is better", MatchOutcome.B_WINS),
        (Language("en"), "Both equal", MatchOutcome.DRAW),
        (Language("de"), "Antwort 2 ist besser", MatchOutcome.B_WINS),
        (Language("en"), "no idea", None),
    ],
)
def test_comparison_grader(language: Language, verdict: str, expected: MatchOutcome | None) -> None:
    judge = ScriptedJudge(json.dumps({"explanation": "because", "better_answer": verdict}))
    output = ComparisonGrader(judge).grade(
        INSTRUCTION, "Only Mike likes Pizza.", "Only Jenny likes Pizza.", language, randomize_order=False
    )

    assert output.outcome == expected
    assert output.reasoning == "because"


@pytest.mark.parametrize("seed", range(4))
def test_comparison_grader_corrects_outcome_for_randomized_order(seed: int) -> None:
    judge = ScriptedJudge(json.dumps({"explanation": "because", "better_answer": "Answer 1 is better"}))
    output = ComparisonGrader(judge).grade(
        INSTRUCTION, "first answer", "second answer", Language("en"), randomize_order=True, seed=seed
    )

    user_prompt = judge.message(Role.USER)
    swapped = user_prompt.index("second answer") < user_prompt.index("first answer")
    assert output.outcome == (MatchOutcome.B_WINS if swapped else MatchOutcome.A_WINS)


def test_comparison_grader_randomizes_order_across_seeds() -> None:
    swaps = set()
    for seed in range(4):
        judge = ScriptedJudge(json.dumps({"explanation": "because", "better_answer": "Answer 1 is better"}))
        ComparisonGrader(judge).grade(
            INSTRUCTION, "first answer", "second answer", Language("en"), randomize_order=True, seed=seed
        )
        user_prompt = judge.message(Role.USER)
        swaps.add(user_prompt.index("second answer") < user_prompt.index("first answer"))
    assert swaps == {True, False}


@pytest.mark.parametrize(
    "language, quality, expected_quality",
    [
        (Language("en"), "A", 5),
        (Language("en"), "C", 3),
        (Language("de"), "gut", 4),
        (Language("fi"), "4", 4),
        (Language("en"), "Z", None),
    ],
)
def test_instruction_grader(language: Language, quality: str, expected_quality: int | None) -> None:
    judge = ScriptedJudge(
        json.dumps(
            {
                "criticism": "fine",
                "quality": quality,
                "is_following_instruction": True,
                "has_correct_grammar_and_spelling": True,
                "is_contradicting_context": False,
                "is_repeating": False,
                "is_deceitful": False,
                "is_harmful": False,
            }
        )
    )
    output = InstructionGrader(judge).grade(INSTRUCTION, COMPLETION, language)

    assert output.criticism == "fine"
    assert output.quality == expected_quality
    assert output.is_following_instruction is True
    assert output.has_correct_grammar_and_spelling is True
    assert output.is_context_consistent is True
    assert output.is_not_repeating is True
    assert output.is_trustworthy is True
    assert output.is_safe is True


def test_instruction_grader_leaves_missing_fields_unset() -> None:
    output = InstructionGrader(ScriptedJudge('{"criticism": "fine"}')).grade(INSTRUCTION, COMPLETION, Language("en"))

    assert output.criticism == "fine"
    assert output.quality is None
    assert output.is_following_instruction is None
    assert output.is_context_consistent is None
    assert output.is_not_repeating is None
    assert output.is_trustworthy is None
    assert output.is_safe is None


@pytest.mark.parametrize(
    "reasons, expected_reasons",
    [
        (["too long", "wrong case"], "• too long\n • wrong case\n "),
        ("too long", "• too long\n "),
        (None, None),
    ],
)
def test_format_correctness_grader(reasons: list[str] | str | None, expected_reasons: str | None) -> None:
    judge = ScriptedJudge(json.dumps({"reasons": reasons, "format_correctness": 1}))
    output = FormatCorrectnessGrader(judge).grade(INSTRUCTION, COMPLETION, Language("en"))

    assert output.reasons == expected_reasons
    assert output.format_correctness == 1


@pytest.mark.parametrize(
    "response, expected",
    [
        ("<coherence_score>85</coherence_score>", 85),
        # the last tag wins, so a score quoted while reasoning does not override the verdict
        ("Reasoning first.\n<coherence_score>10</coherence_score>\n<coherence_score>70</coherence_score>", 70),
        ("42", 42),
    ],
)
def test_coherence_grader(response: str, expected: int) -> None:
    output = CoherenceGrader(ScriptedJudge(response)).grade(INSTRUCTION, COMPLETION, Language("en"))

    assert output.coherence_score == expected


def test_graders_fall_back_to_english_for_unsupported_languages() -> None:
    english_judge = ScriptedJudge('{"format_correctness": 1}')
    other_judge = ScriptedJudge('{"format_correctness": 1}')

    FormatCorrectnessGrader(english_judge).grade(INSTRUCTION, COMPLETION, Language("en"))
    FormatCorrectnessGrader(other_judge).grade(INSTRUCTION, COMPLETION, Language("xx"))

    assert english_judge.prompts == other_judge.prompts
