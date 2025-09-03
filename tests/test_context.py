from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

import eval_framework.context.determined as determined
from eval_framework.context.determined import DeterminedContext
from eval_framework.context.eval import import_models
from eval_framework.context.local import LocalContext
from eval_framework.llm.base import BaseLLM
from eval_framework.llm.models import Llama31_8B_API, Pharia1_7B_Control_API
from eval_framework.tasks.perturbation import PerturbationType


@pytest.fixture
def mock_get_cluster_info_minimal() -> Generator[mock.Mock, None, None]:
    with mock.patch.object(determined, "get_cluster_info") as mock_get_cluster_info:
        mock_info = mock.Mock()
        mock_info.trial = mock.Mock()
        mock_info.trial.hparams = {
            "llm_name": "Llama31_8B_API",
            "output_dir": "dummy",
            "task_args": {
                "num_fewshot": 0,
                "task_name": "ARC",
                "judge_model_name": "Pharia1_7B_Control_API",
                "judge_model_args": {},
            },
        }
        mock_get_cluster_info.return_value = mock_info
        yield mock_get_cluster_info


@pytest.fixture
def mock_get_cluster_info_maximal() -> Generator[mock.Mock, None, None]:
    with mock.patch.object(determined, "get_cluster_info") as mock_get_cluster_info:
        mock_info = mock.Mock()
        mock_info.trial = mock.Mock()
        mock_info.trial.hparams = {
            "llm_name": "Llama31_8B_API",
            "output_dir": "dummy",
            "hf_upload_dir": "hf_dummy",
            "description": "det_description",
            "llm_args": {"dummy_key": "dummy_val"},
            "task_args": {
                "num_fewshot": 0,
                "num_samples": 10,
                "max_tokens": 100,
                "batch_size": 16,
                "task_subjects": ["subject1", "subject2"],
                "task_name": "ARC",
                "judge_model_name": "Pharia1_7B_Control_API",
                "judge_model_args": {},
                "perturbation_config": {
                    "type": "editor",
                    "seed": 123,
                },
            },
        }
        mock_get_cluster_info.return_value = mock_info
        yield mock_get_cluster_info


def test_determined_context_minimal(mock_get_cluster_info_minimal: mock.Mock) -> None:
    # Test that values from determined configuration are used if given but otherwise run.py values are used.
    # Here, we specify as few values as possible in the hparams to test that run.py values are used.
    with DeterminedContext(
        llm_name="some_llm",  # overriden by hparams
        models_path=Path("src/eval_framework/llm/models.py"),
        num_samples=10000,
        max_tokens=111,
        num_fewshot=555,  # overriden by hparams
        task_name="GSM8K",  # overriden by hparams
        task_subjects=None,
        output_dir=Path("dummyXXX"),  # overriden by hparams
        hf_upload_dir="dummy123",
        llm_args={"run_key": "run_val"},
        judge_model_name="Pharia1_7B_Control_API",
        judge_model_args={},
        judge_models_path=Path("src/eval_framework/llm/models.py"),
        batch_size=1,
        description="d",
        perturbation_type="uppercase",
    ) as ctx:
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.hparams is not None
        assert ctx._core_context is not None
        assert ctx.config.llm_class.__name__ == Llama31_8B_API.__name__
        assert ctx.config.num_samples == 10000
        assert ctx.config.max_tokens == 111
        assert ctx.config.num_fewshot == 0
        assert ctx.config.task_name == "ARC"
        assert ctx.config.task_subjects is None
        assert ctx.config.output_dir == Path("dummy")
        assert ctx.config.hf_upload_dir == "dummy123"
        assert ctx.config.llm_args == {"run_key": "run_val"}
        assert ctx.config.llm_judge_class is not None
        assert ctx.config.llm_judge_class.__name__ == Pharia1_7B_Control_API.__name__
        assert ctx.config.judge_model_args is not None
        assert ctx.config.batch_size == 1
        assert ctx.config.description == "d"
        assert ctx.config.perturbation_config is not None
        assert ctx.config.perturbation_config.type == PerturbationType.UPPERCASE
        assert ctx.config.perturbation_config.probability == 0.1  # default
    mock_get_cluster_info_minimal.assert_called()


def test_determined_context_maximal(mock_get_cluster_info_maximal: mock.Mock) -> None:
    # Test that values from determined configuration are used if given but otherwise run.py values are used.
    # Here, we specify as many values as possible in the hparams to test that they are used.
    with DeterminedContext(
        llm_name="some_llm",  # overriden by hparams
        models_path=Path("src/eval_framework/llm/models.py"),
        num_samples=10000,  # overriden by hparams
        max_tokens=111,  # overriden by hparams
        num_fewshot=555,  # overriden by hparams
        task_name="GSM8K",  # overriden by hparams
        task_subjects=None,  # overriden by hparams
        output_dir=Path("dummyXXX"),  # overriden by hparams
        hf_upload_dir="dummy123",  # overriden by hparams
        llm_args={"run_key": "run_val"},  # overriden by hparams
        judge_model_name="Pharia1_7B_Control_API",
        judge_model_args={},
        judge_models_path=Path("src/eval_framework/llm/models.py"),
        batch_size=1,  # overriden by hparams
        description="d",  # overriden by hparams
        perturbation_type="uppercase",  # overriden by hparams
    ) as ctx:
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.hparams is not None
        assert ctx._core_context is not None
        assert ctx.config.llm_class.__name__ == Llama31_8B_API.__name__
        assert ctx.config.num_samples == 10
        assert ctx.config.max_tokens == 100
        assert ctx.config.num_fewshot == 0
        assert ctx.config.task_name == "ARC"
        assert ctx.config.task_subjects == ["subject1", "subject2"]
        assert ctx.config.output_dir == Path("dummy")
        assert ctx.config.hf_upload_dir == "hf_dummy"
        assert ctx.config.llm_args == {"dummy_key": "dummy_val"}
        assert ctx.config.llm_judge_class is not None
        assert ctx.config.llm_judge_class.__name__ == Pharia1_7B_Control_API.__name__
        assert ctx.config.judge_model_args is not None
        assert ctx.config.batch_size == 16
        assert ctx.config.description == "det_description"
        assert ctx.config.perturbation_config is not None
        assert ctx.config.perturbation_config.type == PerturbationType.EDITOR
        assert ctx.config.perturbation_config.probability == 0.1  # default
        assert ctx.config.perturbation_config.seed == 123
    mock_get_cluster_info_maximal.assert_called()


def test_local_context() -> None:
    with LocalContext(
        llm_name="Llama31_8B_API",
        models_path=Path("src/eval_framework/llm/models.py"),
        num_samples=10,
        num_fewshot=0,
        task_name="ARC",
        output_dir=Path("dummy"),
        hf_upload_dir="dummy22",
        llm_args={"dummy": "dummy"},
        judge_model_name="Pharia1_7B_Control_API",
        judge_model_args={},
        judge_models_path=Path("src/eval_framework/llm/models.py"),
        batch_size=1,
    ) as ctx:
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.config.llm_class.__name__ == Llama31_8B_API.__name__
        assert ctx.config.num_samples == 10
        assert ctx.config.num_fewshot == 0
        assert ctx.config.task_name == "ARC"
        assert ctx.config.output_dir == Path("dummy")
        assert ctx.config.hf_upload_dir == "dummy22"
        assert ctx.config.llm_args == {"dummy": "dummy"}
        assert ctx.config.llm_judge_class is not None
        assert ctx.config.llm_judge_class.__name__ == Pharia1_7B_Control_API.__name__
        assert ctx.config.judge_model_args is not None


def test_import_models() -> None:
    models = import_models(Path("src/eval_framework/llm/models.py"))
    huggingface_llm = import_models(Path("src/eval_framework/llm/huggingface.py"))

    assert "SmolLM135M" in models
    assert "Pythia410m" in models
    assert "HFLLM" in huggingface_llm
    del huggingface_llm["HFLLM"]

    for model in models.values():
        assert issubclass(model, BaseLLM)


def test_fail_validation_when_required_judge_not_given() -> None:
    with pytest.raises(ValidationError):
        with LocalContext(
            llm_name="Llama31_8B_API",
            models_path=Path("src/eval_framework/llm/models.py"),
            num_samples=10,
            num_fewshot=0,
            task_name="EvaluationSuiteConciseness",  # requires a judge
            output_dir=Path("dummy"),
            hf_upload_dir="dummy22",
            llm_args={"dummy": "dummy"},
            judge_model_name=None,  # but it's not given
            judge_model_args={},
            judge_models_path=Path("src/eval_framework/llm/models.py"),
            batch_size=1,
        ) as _:
            pass
