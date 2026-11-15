import pytest

from eval_framework.metrics.llm.graders.language import Language, LanguageNotSupportedError
from eval_framework.metrics.llm.graders.models import parse_json_output


@pytest.mark.parametrize(
    "output, expected",
    [
        ('{"answer": "yes"}', {"answer": "yes"}),
        ('Sure, here it is:\n```json\n{"answer": "no"}\n```', {"answer": "no"}),
        ("no json at all", {}),
        ('{"answer": yes}', {}),
    ],
)
def test_parse_json_output_extracts_the_object_or_falls_back_to_empty(output: str, expected: dict[str, str]) -> None:
    assert parse_json_output(output) == expected


def test_language_config_returns_the_config_of_the_language() -> None:
    configs = {Language("en"): "english prompt", Language("de"): "german prompt"}

    assert Language("de").language_config(configs) == "german prompt"


def test_language_config_rejects_a_language_without_config() -> None:
    configs = {Language("en"): "english prompt"}

    with pytest.raises(LanguageNotSupportedError, match=r"fr not in \(en\)"):
        Language("fr").language_config(configs)
