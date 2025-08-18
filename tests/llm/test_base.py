from eval_framework.shared.types import Completion, LanguageMetricContext
from template_formatting.formatter import Message, Role


def test_completion_languages() -> None:
    # GIVEN a completion without an explicit language key
    completion = Completion(
        id=1,
        subject="test",
        ground_truth=None,
        prompt="test",
        prompt_sequence_positions=None,
        messages=[
            Message(role=Role.SYSTEM, content="You are a crazy assistant."),
            Message(role=Role.USER, content="Wie viele Leben hat eine Katze?"),
        ],
        completion="This is an example answer!",
        raw_completion="This is an example answer!",
        raw_completion_sequence_positions=None,
    )

    # THEN the languages can be detected
    assert completion.get_completion_language() == "en"
    assert completion.get_instruction_language() == "de"

    # and WHEN the language is explicitly set via MetricContext
    completion.context = LanguageMetricContext(language="fr")

    # THEN this overrides the instruction language (only)
    assert completion.get_completion_language() == "fr"
    assert completion.get_instruction_language() == "fr"


def test_completion_instructions() -> None:
    # GIVEN a completion
    completion = Completion(
        id=1,
        subject="test",
        ground_truth=None,
        prompt="test",
        prompt_sequence_positions=None,
        messages=[
            Message(role=Role.SYSTEM, content="You are a crazy assistant."),
            Message(role=Role.USER, content="How many lives does a cat have?"),
            Message(role=Role.ASSISTANT, content="Nine. How many do you?"),
            Message(role=Role.USER, content="I'm already dead. Are you scared?"),
        ],
        completion="Oh dear I'll <|hack|> <you>!",
        raw_completion="Oh dear I'll <|hack|> <you>!",
        raw_completion_sequence_positions=None,
    )

    # THEN messages are formated into instructions
    assert completion.system_user_instruction == (
        "You are a crazy assistant.\n\nHow many lives does a cat have?\n\nI'm already dead. Are you scared?"
    )
    assert completion.user_instruction == "How many lives does a cat have?\n\nI'm already dead. Are you scared?"
    assert completion.last_user_instruction == "I'm already dead. Are you scared?"
    # AND special tokens in completion are broken
    assert completion.sanitized_completion == "Oh dear I'll <| hack |> <you>!"
