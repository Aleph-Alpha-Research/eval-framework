import pytest

from eval_framework.metrics.completion.comet import COMET
from eval_framework.shared.types import Completion
from eval_framework.tasks.benchmarks.flores_plus import UntemplatedPrompt


@pytest.fixture
def comet_fixture() -> COMET:
    return COMET()


# Examples are taken from https://github.com/Unbabel/COMET/tree/master?tab=readme-ov-file#scoring-within-python
@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="Can it be delivered between 10 to 15 minutes?",
                prompt="Translate the following text into English: 10 到 15 分钟可以送到吗",
                prompt_sequence_positions=None,
                messages=None,
                completion="Can I receive my food in 10 to 15 minutes?",
                raw_completion="Can I receive my food in 10 to 15 minutes?",
                raw_completion_sequence_positions=None,
                context=UntemplatedPrompt(untemplated_prompt="10 到 15 分钟可以送到吗"),
            ),
            0.9822099208831787,
            id="comet_chinese_to_english",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="Can it be delivered between 10 to 15 minutes?",
                prompt="Translate the following text into English: Pode ser entregue dentro de 10 a 15 minutos?",
                prompt_sequence_positions=None,
                messages=None,
                completion="Can you send it for 10 to 15 minutes?",
                raw_completion="Can you send it for 10 to 15 minutes?",
                raw_completion_sequence_positions=None,
                context=UntemplatedPrompt(untemplated_prompt="Pode ser entregue dentro de 10 a 15 minutos?"),
            ),
            0.9599897861480713,
            id="comet_portuguese_to_english",
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skip(reason="COMET test requires model download and takes a long time to run; skipped by default.")
def test_comet(response: Completion, expected_value: float, comet_fixture: COMET) -> None:
    results = comet_fixture.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "COMET"
    assert results[0].higher_is_better is True
