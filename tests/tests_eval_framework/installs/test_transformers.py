def test_transformers_import() -> None:
    from eval_framework.llm.huggingface import HFLLM

    assert HFLLM is not None
