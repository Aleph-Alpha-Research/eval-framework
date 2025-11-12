def test_core_import() -> None:
    from eval_framework.llm.base import BaseLLM

    assert BaseLLM is not None
    from eval_framework.tasks.base import BaseTask

    assert BaseTask is not None
    from template_formatting.formatter import BaseFormatter

    assert BaseFormatter is not None
