"""
Tests for EvalConfig validation logic, specifically the ast.literal_eval() functionality.
"""

from eval_framework.llm.vllm import Qwen3_0_6B_VLLM
from eval_framework.tasks.eval_config import EvalConfig


class TestEvalConfigLLMArgsValidation:
    """Test the llm_args validation with ast.literal_eval() functionality."""

    def test_int_conversion(self) -> None:
        """Test string to int conversion."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {"max_tokens": "42", "batch_size": "10"},
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["max_tokens"] == 42
        assert isinstance(config.llm_args["max_tokens"], int)
        assert config.llm_args["batch_size"] == 10
        assert isinstance(config.llm_args["batch_size"], int)

    def test_float_conversion(self) -> None:
        """Test string to float conversion."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {"temperature": "0.7", "top_p": "0.95", "learning_rate": "1e-4"},
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["temperature"] == 0.7
        assert isinstance(config.llm_args["temperature"], float)
        assert config.llm_args["top_p"] == 0.95
        assert isinstance(config.llm_args["top_p"], float)
        assert config.llm_args["learning_rate"] == 1e-4
        assert isinstance(config.llm_args["learning_rate"], float)

    def test_boolean_conversion(self) -> None:
        """Test string to boolean conversion."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {"use_cache": "True", "verbose": "False"},
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["use_cache"] is True
        assert isinstance(config.llm_args["use_cache"], bool)
        assert config.llm_args["verbose"] is False
        assert isinstance(config.llm_args["verbose"], bool)

    def test_none_conversion(self) -> None:
        """Test string to None conversion."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {"seed": "None", "checkpoint": "None"},
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["seed"] is None
        assert config.llm_args["checkpoint"] is None

    def test_list_conversion(self) -> None:
        """Test string to list conversion."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {
                "stop_tokens": '["<|end|>", "\\n", "END"]',
                "numbers": "[1, 2, 3, 4]",
                "mixed_list": '["text", 42, 3.14, True, None]',
            },
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["stop_tokens"] == ["<|end|>", "\n", "END"]
        assert isinstance(config.llm_args["stop_tokens"], list)
        assert config.llm_args["numbers"] == [1, 2, 3, 4]
        assert isinstance(config.llm_args["numbers"], list)
        assert config.llm_args["mixed_list"] == ["text", 42, 3.14, True, None]
        assert isinstance(config.llm_args["mixed_list"], list)

    def test_dict_conversion(self) -> None:
        """Test string to dict conversion."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {
                "config": '{"temperature": 0.8, "max_tokens": 100}',
                "nested": '{"outer": {"inner": 42, "flag": True}}',
            },
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["config"] == {"temperature": 0.8, "max_tokens": 100}
        assert isinstance(config.llm_args["config"], dict)
        assert config.llm_args["nested"] == {"outer": {"inner": 42, "flag": True}}
        assert isinstance(config.llm_args["nested"], dict)

    def test_string_preservation(self) -> None:
        """Test that regular strings are preserved as strings."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {
                "model_name": "gpt-4",
                "api_key": "sk-1234567890",
                "endpoint": "https://api.openai.com",
                "malformed_list": "[1, 2, 3",  # Invalid syntax
                "malformed_dict": '{"key": }',  # Invalid syntax
            },
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["model_name"] == "gpt-4"
        assert isinstance(config.llm_args["model_name"], str)
        assert config.llm_args["api_key"] == "sk-1234567890"
        assert isinstance(config.llm_args["api_key"], str)
        assert config.llm_args["endpoint"] == "https://api.openai.com"
        assert isinstance(config.llm_args["endpoint"], str)
        # Malformed literals should remain as strings
        assert config.llm_args["malformed_list"] == "[1, 2, 3"
        assert isinstance(config.llm_args["malformed_list"], str)
        assert config.llm_args["malformed_dict"] == '{"key": }'
        assert isinstance(config.llm_args["malformed_dict"], str)

    def test_nested_dict_recursive_conversion(self) -> None:
        """Test that nested dictionaries are converted recursively."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {
                "sampling_params": {
                    "temperature": "0.7",
                    "top_p": "0.9",
                    "max_tokens": "100",
                    "stop": '["<|end|>", "\\n"]',
                    "use_beam_search": "False",
                    "seed": "None",
                },
                "model_config": {
                    "trust_remote_code": "True",
                    "gpu_memory_utilization": "0.8",
                    "max_model_len": "2048",
                    "nested_config": {
                        "deep_value": "42",
                        "deep_flag": "True",
                    },
                },
            },
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        # Test sampling_params conversion
        sp = config.llm_args["sampling_params"]
        assert sp["temperature"] == 0.7
        assert isinstance(sp["temperature"], float)
        assert sp["top_p"] == 0.9
        assert isinstance(sp["top_p"], float)
        assert sp["max_tokens"] == 100
        assert isinstance(sp["max_tokens"], int)
        assert sp["stop"] == ["<|end|>", "\n"]
        assert isinstance(sp["stop"], list)
        assert sp["use_beam_search"] is False
        assert isinstance(sp["use_beam_search"], bool)
        assert sp["seed"] is None

        # Test model_config conversion
        mc = config.llm_args["model_config"]
        assert mc["trust_remote_code"] is True
        assert isinstance(mc["trust_remote_code"], bool)
        assert mc["gpu_memory_utilization"] == 0.8
        assert isinstance(mc["gpu_memory_utilization"], float)
        assert mc["max_model_len"] == 2048
        assert isinstance(mc["max_model_len"], int)

        # Test deeply nested conversion
        nc = mc["nested_config"]
        assert nc["deep_value"] == 42
        assert isinstance(nc["deep_value"], int)
        assert nc["deep_flag"] is True
        assert isinstance(nc["deep_flag"], bool)

    def test_already_correct_types_preserved(self) -> None:
        """Test that values with correct types are not modified."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {
                "already_int": 42,
                "already_float": 3.14,
                "already_bool": True,
                "already_none": None,
                "already_list": [1, 2, 3],
                "already_dict": {"key": "value"},
                "mixed": {
                    "string_to_convert": "123",
                    "already_converted": 456,
                },
            },
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["already_int"] == 42
        assert isinstance(config.llm_args["already_int"], int)
        assert config.llm_args["already_float"] == 3.14
        assert isinstance(config.llm_args["already_float"], float)
        assert config.llm_args["already_bool"] is True
        assert isinstance(config.llm_args["already_bool"], bool)
        assert config.llm_args["already_none"] is None
        assert config.llm_args["already_list"] == [1, 2, 3]
        assert isinstance(config.llm_args["already_list"], list)
        assert config.llm_args["already_dict"] == {"key": "value"}
        assert isinstance(config.llm_args["already_dict"], dict)

        # Test mixed scenario
        mixed = config.llm_args["mixed"]
        assert mixed["string_to_convert"] == 123
        assert isinstance(mixed["string_to_convert"], int)
        assert mixed["already_converted"] == 456
        assert isinstance(mixed["already_converted"], int)

    def test_edge_cases(self) -> None:
        """Test edge cases and potential problematic inputs."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_args": {
                "empty_string": "",
                "whitespace": "   ",
                "zero": "0",
                "negative": "-42",
                "scientific_notation": "1e-5",
                "large_number": "999999999999999",
                "empty_list": "[]",
                "empty_dict": "{}",
                "single_quote_string": "'hello'",
                "double_quote_string": '"world"',
            },
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.llm_args["empty_string"] == ""
        assert isinstance(config.llm_args["empty_string"], str)
        assert config.llm_args["whitespace"] == "   "
        assert isinstance(config.llm_args["whitespace"], str)
        assert config.llm_args["zero"] == 0
        assert isinstance(config.llm_args["zero"], int)
        assert config.llm_args["negative"] == -42
        assert isinstance(config.llm_args["negative"], int)
        assert config.llm_args["scientific_notation"] == 1e-5
        assert isinstance(config.llm_args["scientific_notation"], float)
        assert config.llm_args["large_number"] == 999999999999999
        assert isinstance(config.llm_args["large_number"], int)
        assert config.llm_args["empty_list"] == []
        assert isinstance(config.llm_args["empty_list"], list)
        assert config.llm_args["empty_dict"] == {}
        assert isinstance(config.llm_args["empty_dict"], dict)
        assert config.llm_args["single_quote_string"] == "hello"
        assert isinstance(config.llm_args["single_quote_string"], str)
        assert config.llm_args["double_quote_string"] == "world"
        assert isinstance(config.llm_args["double_quote_string"], str)


class TestEvalConfigJudgeModelArgsValidation:
    """Test the judge_model_args validation (which still uses the old approach)."""

    def test_judge_model_args_conversion(self) -> None:
        """Test that judge_model_args still uses the old conversion approach."""
        config_data = {
            "llm_class": Qwen3_0_6B_VLLM,
            "llm_judge_class": Qwen3_0_6B_VLLM,
            "judge_model_args": {
                "temperature": "0.7",
                "max_tokens": "100",
                "model_name": "gpt-4",  # String should remain string
                "use_cache": "True",  # String should remain string (not converted to bool)
            },
            "task_name": "MMLU",
        }

        config = EvalConfig(**config_data)

        assert config.judge_model_args["temperature"] == 0.7
        assert isinstance(config.judge_model_args["temperature"], float)
        assert config.judge_model_args["max_tokens"] == 100
        assert isinstance(config.judge_model_args["max_tokens"], int)
        assert config.judge_model_args["model_name"] == "gpt-4"
        assert isinstance(config.judge_model_args["model_name"], str)
        # Note: The old approach doesn't convert booleans
        assert config.judge_model_args["use_cache"] == "True"
        assert isinstance(config.judge_model_args["use_cache"], str)
