from unittest.mock import patch

from eval_framework.llm.huggingface import HFLLM


def test_transformers_import() -> None:
    # Mock the __init__ method to avoid actual initialization
    with patch.object(HFLLM, "__init__", lambda self, model_name: None):
        # Test HuggingFace LLM import
        model = HFLLM(model_name="gpt2")  # type: ignore

        # Check basic expectations
        assert isinstance(model, HFLLM)
        assert hasattr(model, "__class__")


def main() -> None:
    test_transformers_import()


if __name__ == "__main__":
    main()
