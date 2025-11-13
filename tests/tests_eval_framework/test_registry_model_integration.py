from pathlib import Path
from unittest.mock import patch

import pytest
import wandb

from eval_framework import main as main_file
from eval_framework.llm.huggingface import HFLLMRegistryModel
from eval_framework.llm.vllm import VLLMRegistryModel
from eval_framework.tasks.benchmarks.arc import ARC
from eval_framework.tasks.eval_config import EvalConfig
from tests.tests_eval_framework.conftest import MockLLM


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


def test_additional_artifact_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wandb_run: wandb.Run) -> None:
    """Test that additional wandb artifacts are registered as being used during evaluation."""
    mock_llm = MockLLM()
    eval_config = EvalConfig(
        task_name=ARC.NAME,
        num_fewshot=0,
        num_samples=10,
        output_dir=tmp_path,
        llm_class=mock_llm.__class__,
    )

    monkeypatch.setenv("WANDB_ADDITIONAL_ARTIFACT_REFERENCES", "artifact1:latest, artifact2:v2.0.0")
    with patch.object(main_file.wandb, "use_artifact") as mock_use_artifact:
        main_file.main(mock_llm, eval_config)
    mock_use_artifact.assert_any_call("artifact1:latest")
    mock_use_artifact.assert_any_call("artifact2:v2.0.0")

    monkeypatch.delenv("WANDB_ADDITIONAL_ARTIFACT_REFERENCES", raising=False)
    with patch.object(main_file.wandb, "use_artifact") as mock_use_artifact:
        main_file.main(mock_llm, eval_config)
    mock_use_artifact.assert_not_called()
