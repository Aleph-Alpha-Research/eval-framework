def test_vllm_import() -> None:
    from eval_framework.llm.vllm import VLLMModel

    assert VLLMModel is not None
