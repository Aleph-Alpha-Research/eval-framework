from unittest.mock import patch

from eval_framework.context.determined import DeterminedContext
from eval_framework.llm.aleph_alpha import AlephAlphaAPIModel
from eval_framework.llm.huggingface import HFLLM
from eval_framework.llm.openai import OpenAIModel
from eval_framework.tasks.task_names import registered_tasks_iter


def test_all_import() -> None:
    # Mock the __init__ methods to avoid actual initialization
    with patch.object(DeterminedContext, "__init__", lambda self: None):
        # Test Determined context import
        context = DeterminedContext()  # type: ignore

        # Check basic expectations
        assert isinstance(context, DeterminedContext)
        assert hasattr(context, "__class__")

    with patch.object(HFLLM, "__init__", lambda self, model_name: None):
        hf_model = HFLLM(model_name="gpt2")  # type: ignore

        assert isinstance(hf_model, HFLLM)
        assert hasattr(hf_model, "__class__")

    with patch.object(AlephAlphaAPIModel, "__init__", lambda self, model_name: None):
        api_model = AlephAlphaAPIModel(model_name="llama-3-8b-instruct")  # type: ignore

        assert isinstance(api_model, AlephAlphaAPIModel)
        assert hasattr(api_model, "__class__")

    with patch.object(OpenAIModel, "__init__", lambda self, model_name: None):
        openai_model = OpenAIModel(model_name="gpt-3.5-turbo")  # type: ignore

        assert isinstance(openai_model, OpenAIModel)
        assert hasattr(openai_model, "__class__")

    # Test task registry import
    for _ in registered_tasks_iter():
        pass


def main() -> None:
    test_all_import()


if __name__ == "__main__":
    main()
