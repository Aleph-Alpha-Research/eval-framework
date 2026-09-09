import pytest

from eval_framework.tasks.benchmarks.ifeval import IFEval
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_ifeval_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding ifeval tasks
_ifeval_registry = Registry()
register_ifeval_tasks(registry=_ifeval_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _ifeval_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_ifeval_registry)


def test_get_context_gives_same_kwargs_whether_absent_keys_are_omitted_or_none() -> None:
    """
    Each IFEval sample has a list of kwargs dicts, one per instruction. The HF `datasets` library decides how a
    dict with absent keys is loaded: version 5 omits them, older versions (and caches written by them) include every
    key with None. The task must produce the same context from both, otherwise a run's result depends on which
    `datasets` version wrote the cache.
    """
    absent_keys_omitted = [{"keyword": "knock", "frequency": 2}, {"keywords": ["cat"]}]
    absent_keys_as_none = [
        {"keyword": "knock", "frequency": 2, "keywords": None},
        {"keyword": None, "frequency": None, "keywords": ["cat"]},
    ]
    task = IFEval()

    for kwargs in (absent_keys_omitted, absent_keys_as_none):
        item = {"key": 142, "prompt": "", "instruction_id_list": [], "kwargs": kwargs}
        assert task._get_context(item).additional_kwargs == absent_keys_omitted


def test_get_context_casts_integral_floats_to_int_and_rejects_others() -> None:
    """Some dataset variants type integer kwargs as float; a non-integer value would be silently truncated."""
    task = IFEval()
    item = {"key": 0, "prompt": "", "instruction_id_list": [], "kwargs": [{"frequency": 2.0}]}
    assert task._get_context(item).additional_kwargs == [{"frequency": 2}]

    item["kwargs"] = [{"frequency": 2.5}]
    with pytest.raises(AssertionError):
        task._get_context(item)
