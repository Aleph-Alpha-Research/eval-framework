import pytest

from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_global_mmlu_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, NoStripConcatFormatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding global_mmlu tasks
_global_mmlu_registry = Registry()
register_global_mmlu_tasks(registry=_global_mmlu_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _global_mmlu_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_global_mmlu_registry)
