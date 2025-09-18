from pathlib import Path

import pytest

from eval_framework.llm.huggingface import HFLLMRegistryModel
from eval_framework.llm.vllm import VLLMRegistryModel
from eval_framework.tasks.eval_config import EvalConfig


@pytest.mark.parametrize(
    "model_class,extra_args",
    [
        pytest.param(HFLLMRegistryModel, {}, id="hfllm_backend"),
        pytest.param(
            VLLMRegistryModel,
            {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.8, "batch_size": 4},
            id="vllm_backend_with_params",
        ),
    ],
)
def test_registry_model_config_integration(model_class: type, extra_args: dict[str, int | float]) -> None:
    base_llm_args = {
        "artifact_name": "test-model",
        "version": "v1.0.0",
        "formatter": "ConcatFormatter",
    }

    llm_args = {**base_llm_args, **extra_args}

    config = EvalConfig(
        llm_class=model_class,
        llm_args=llm_args,
        task_name="ARC",
        num_fewshot=2,
        num_samples=10,
        output_dir=Path("/tmp/test_output"),
    )

    assert config.llm_class == model_class

    if model_class == VLLMRegistryModel:
        assert config.llm_args["tensor_parallel_size"] == 2
        assert config.llm_args["gpu_memory_utilization"] == 0.8
        assert config.llm_args["batch_size"] == 4
