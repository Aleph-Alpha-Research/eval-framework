import pytest

from eval_framework.tasks.benchmarks.mmlu_pro import MMLU_PRO_COT_V2
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_mmlu_pro_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, NoStripConcatFormatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding mmlu_pro tasks
_mmlu_pro_registry = Registry()
register_mmlu_pro_tasks(registry=_mmlu_pro_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _mmlu_pro_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_mmlu_pro_registry)


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("Therefore, the answer is (J)", "J"),
        ("Therefore, the answer is J", "J"),
        ("**Therefore, the answer is (h).**", "H"),
        ("The answer is (A). Wait, actually the answer is (C).", "C"),
        ("Question: what is 2+2?", "[invalid]"),
        ("", "[invalid]"),
    ],
)
def test_extract_answer(completion: str, expected: str) -> None:
    assert MMLU_PRO_COT_V2().post_process_generated_completion(completion) == expected
