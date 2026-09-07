import pytest

from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_humaneval_plus_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, NoStripConcatFormatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding humaneval_plus tasks
_humaneval_plus_registry = Registry()
register_humaneval_plus_tasks(registry=_humaneval_plus_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _humaneval_plus_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, num_fewshot=0, registry=_humaneval_plus_registry)
