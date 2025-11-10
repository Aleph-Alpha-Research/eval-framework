from unittest.mock import patch

from eval_framework.llm.mistral import MistralVLLM


def test_mistral_import() -> None:
    # Mock the __init__ method to avoid actual initialization
    with patch.object(MistralVLLM, "__init__", lambda self, model_name: None):
        # Test Mistral VLLM import
        model = MistralVLLM(model_name="mistral-7b-instruct-v0.1")

        # Check basic expectations
        assert isinstance(model, MistralVLLM)
        assert hasattr(model, "__class__")


def main() -> None:
    test_mistral_import()


if __name__ == "__main__":
    main()
