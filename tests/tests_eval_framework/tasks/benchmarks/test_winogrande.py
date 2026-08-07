import pytest

from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_winogrande_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding winogrande tasks
_winogrande_registry = Registry()
register_winogrande_tasks(registry=_winogrande_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _winogrande_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_winogrande_registry)
