from pathlib import Path

import pytest

from eval_framework.llm.models import RegistryModel
from eval_framework.task_names import TaskName
from eval_framework.tasks.eval_config import EvalConfig


@pytest.mark.xfail()
def test_registry_model_backend_selection() -> None:
    RegistryModel(artifact_name="test-model", backend="invalid_backend")


@pytest.mark.parametrize(
    "backend,extra_args,expected_backend",
    [
        pytest.param("hfllm", {}, "hfllm", id="hfllm_backend"),
        pytest.param(
            "vllm",
            {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.8, "batch_size": 4},
            "vllm",
            id="vllm_backend_with_params",
        ),
    ],
)
def test_registry_model_config_integration(backend, extra_args, expected_backend) -> None:
    base_llm_args = {
        "artifact_name": "test-model",
        "version": "v1.0.0",
        "formatter": "ConcatFormatter",
        "backend": backend,
    }

    llm_args = {**base_llm_args, **extra_args}

    config = EvalConfig(
        llm_class=RegistryModel,
        llm_args=llm_args,
        task_name=TaskName.ARC,
        num_fewshot=2,
        num_samples=10,
        output_dir=Path("/tmp/test_output"),
    )

    assert config.llm_class == RegistryModel
    assert config.llm_args["backend"] == expected_backend

    if backend == "vllm":
        assert config.llm_args["tensor_parallel_size"] == 2
        assert config.llm_args["gpu_memory_utilization"] == 0.8
        assert config.llm_args["batch_size"] == 4
