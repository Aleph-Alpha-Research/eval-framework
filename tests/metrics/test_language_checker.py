import pytest

from eval_framework.exceptions import LogicError
from eval_framework.metrics.completion.language_checker import (
    LanguageChecker,
    LanguageConsistencyChecker,
    LanguageRawConsistencyChecker,
)
from eval_framework.shared.types import Completion
from template_formatting.formatter import Message, Role


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="en",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="The sixth sick sheik's sixth sheep's sick.",
                raw_completion="The sixth sick sheik's sixth sheep's sick.",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_en_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="en",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                # completion="Lorem ipsum dolor sit amet", # lingua thinks it's en, latin is not in AVAILABLE_LANGUAGES
                completion="Un chasseur sachant chasser sait chasser sans son chien",
                raw_completion="Un chasseur sachant chasser sait chasser sans son chien",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_en_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="de",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Brautkleid bleibt Brautkleid und Blaukraut bleibt Blaukraut",
                raw_completion="Brautkleid bleibt Brautkleid und Blaukraut bleibt Blaukraut",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_de_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="de",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Lorem ipsum dolor sit amet",
                raw_completion="Lorem ipsum dolor sit amet",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_de_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="fr",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Un chasseur sachant chasser sait chasser sans son chien",
                raw_completion="Un chasseur sachant chasser sait chasser sans son chien",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_fr_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="fr",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Lorem ipsum dolor sit amet",
                raw_completion="Lorem ipsum dolor sit amet",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_fr_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="es",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Mi mamá me mima mucho",
                raw_completion="Mi mamá me mima mucho",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_es_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="es",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Lorem ipsum dolor sit amet",
                raw_completion="Lorem ipsum dolor sit amet",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_es_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="it",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Se l'arcivescovo di Costantinopoli si disarcivescoviscostantinopolizzasse",
                raw_completion="Se l'arcivescovo di Costantinopoli si disarcivescoviscostantinopolizzasse",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_it_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="it",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Lorem ipsum dolor sit amet",
                raw_completion="Lorem ipsum dolor sit amet",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_it_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="nl",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Lientje leerde Lotje lopen langs de lange Lindenlaan",
                raw_completion="Lientje leerde Lotje lopen langs de lange Lindenlaan",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_nl_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="nl",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Lorem ipsum dolor sit amet",
                raw_completion="Lorem ipsum dolor sit amet",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_nl_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="pt",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Um prato de trigo para três tigres",
                raw_completion="Um prato de trigo para três tigres",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_pt_completion",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="pt",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="De koetsier poetst de postkoets",
                raw_completion="De koetsier poetst de postkoets",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_pt_completion",
        ),
    ],
)
def test_language_checker(response: Completion, expected_value: float) -> None:
    metric = LanguageChecker()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Language Check"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "completion",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="xx",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="A",
                raw_completion="A",
                raw_completion_sequence_positions=None,
            ),
            id="unknown_language",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="no",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="A",
                raw_completion="A",
                raw_completion_sequence_positions=None,
            ),
            id="unavailable_language",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="A",
                raw_completion="A",
                raw_completion_sequence_positions=None,
            ),
            id="missing_ground_truth",
        ),
    ],
)
def test_language_checker_errors(completion: Completion) -> None:
    metric = LanguageChecker()
    with pytest.raises(LogicError):
        metric.calculate(completion)


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=[Message(role=Role.USER, content="Hallo, erzähl mir etwas!")],
                completion="Brautkleid bleibt Brautkleid und Blaukraut bleibt Blaukraut",
                raw_completion="Brautkleid bleibt Brautkleid und Blaukraut bleibt Blaukraut",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="matching_language",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=[Message(role=Role.USER, content="Hallo, erzähl mir etwas!")],
                completion="De koetsier poetst de postkoets",
                raw_completion="De koetsier poetst de postkoets",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="not_matching_language",
        ),
    ],
)
def test_language_consistency_checker(response: Completion, expected_value: float) -> None:
    metric = LanguageConsistencyChecker()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Language Consistency"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=[Message(role=Role.USER, content="Hallo, erzähl mir etwas!")],
                completion="Brautkleid bleibt Brautkleid und Blaukraut bleibt Blaukraut",
                raw_completion="Brautkleid bleibt Brautkleid und Blaukraut bleibt Blaukraut",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="matching_language",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=[Message(role=Role.USER, content="Hallo, erzähl mir etwas!")],
                completion="De koetsier poetst de postkoets",
                raw_completion="De koetsier poetst de postkoets",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="not_matching_language",
        ),
    ],
)
def test_language_raw_consistency_checker(response: Completion, expected_value: float) -> None:
    metric = LanguageRawConsistencyChecker()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Language Consistency Raw"
    assert results[0].higher_is_better is True
