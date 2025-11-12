def test_openai_import() -> None:
    from eval_framework.llm.openai import OpenAIModel

    assert OpenAIModel is not None
