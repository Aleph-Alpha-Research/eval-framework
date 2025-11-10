from unittest.mock import patch

from eval_framework.llm.vllm import VLLMModel


def test_vllm_import() -> None:
    # Mock the __init__ method to avoid actual initialization
    with patch.object(VLLMModel, "__init__", lambda self, model_name: None) as mock_init:  # type: ignore
        # Test VLLM Model import
        model = VLLMModel(model_name="vllm-test-model")

        # Check basic expectations
        assert isinstance(model, VLLMModel)
        assert hasattr(model, "__class__")


def main() -> None:
    test_vllm_import()


if __name__ == "__main__":
    main()
