def test_core_import() -> None:
    from eval_framework.llm.base import BaseLLM

    assert BaseLLM is not None
    # Importing the registry initializes eval_framework.tasks, which registers every benchmark —
    # so this import proves the whole task catalog builds in a core (no-extras) install.
    from eval_framework.tasks.registry import registered_task_names

    assert len(registered_task_names()) > 0
    from template_formatting.formatter import BaseFormatter

    assert BaseFormatter is not None

    from eval_framework.llm.openai import OpenAIModel

    assert OpenAIModel is not None
