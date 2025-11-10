from unittest.mock import patch

from eval_framework.llm.aleph_alpha import AlephAlphaAPIModel


def test_aleph_alpha_import() -> None:
    # Mock the __init__ method to avoid actual initialization
    with patch.object(AlephAlphaAPIModel, "__init__", lambda self, model_name: None):  # type: ignore
        # Test Aleph Alpha API Model import
        model = AlephAlphaAPIModel(model_name="llama-3-8b-instruct")

        # Check basic expectations
        assert isinstance(model, AlephAlphaAPIModel)
        assert hasattr(model, "__class__")


def main() -> None:
    test_aleph_alpha_import()


if __name__ == "__main__":
    main()
