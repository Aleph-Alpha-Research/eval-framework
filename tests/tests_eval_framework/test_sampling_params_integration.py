"""
Tests for sampling parameters integration across CLI, configs, and model initialization.
"""

from unittest.mock import patch

from eval_framework.llm.openai import OpenAIModel
from eval_framework.run import parse_args
from eval_framework.tasks.eval_config import EvalConfig


def test_cli_nested_sampling_params_parsing() -> None:
    """Test CLI args with nested sampling_params get parsed correctly."""
    test_args = [
        "run.py",
        "--llm-name",
        "Qwen3_0_6B_VLLM",
        "--task-name",
        "MMLU",
        "--llm-args",
        "max_model_len=128",
        "sampling_params.temperature=0.8",
        "sampling_params.top_p=0.95",
    ]

    with patch("sys.argv", test_args):
        args = parse_args()

    expected_llm_args = {"max_model_len": "128", "sampling_params": {"temperature": "0.8", "top_p": "0.95"}}

    assert args.llm_args == expected_llm_args


def test_eval_config_validates_nested_sampling_params() -> None:
    """Test EvalConfig correctly validates and converts nested sampling_params."""
    config_data = {
        "llm_class": OpenAIModel,
        "llm_args": {
            "max_model_len": "128",
            "sampling_params": {"temperature": "0.7", "top_p": "0.9", "max_tokens": "100"},
        },
        "task_name": "MMLU",
        "num_fewshot": 5,
        "num_samples": 10,
    }

    config = EvalConfig(**config_data)

    # Verify recursive type conversion works
    assert config.llm_args["max_model_len"] == 128  # int conversion
    assert config.llm_args["sampling_params"]["temperature"] == 0.7  # float conversion
    assert config.llm_args["sampling_params"]["top_p"] == 0.9  # float conversion
    assert config.llm_args["sampling_params"]["max_tokens"] == 100  # int conversion


def test_multiple_nested_keys() -> None:
    """Test parsing multiple levels of nesting in CLI args."""
    test_args = [
        "run.py",
        "--llm-name",
        "test",
        "--llm-args",
        "sampling_params.temperature=0.5",
        "sampling_params.stop=END",
        "other_config.nested_value=42",
    ]

    with patch("sys.argv", test_args):
        args = parse_args()

    expected_llm_args = {
        "sampling_params": {"temperature": "0.5", "stop": "END"},
        "other_config": {"nested_value": "42"},
    }

    assert args.llm_args == expected_llm_args
