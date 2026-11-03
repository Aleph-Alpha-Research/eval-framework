def test_core_import() -> None:
    from eval_framework.llm.base import BaseLLM

    assert BaseLLM is not None
    from eval_framework.composed import ComposedBenchmark

    assert ComposedBenchmark is not None
    from template_formatting.formatter import BaseFormatter

    assert BaseFormatter is not None

    from eval_framework.llm.openai import OpenAIModel

    assert OpenAIModel is not None
