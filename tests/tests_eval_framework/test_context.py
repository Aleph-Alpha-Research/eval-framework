from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest

import eval_framework.context.determined as determined
from eval_framework.context.determined import DeterminedContext
from eval_framework.context.local import LocalContext
from eval_framework.llm.openai import OpenAIModel


@pytest.fixture
def mock_get_cluster_info_minimal() -> Generator[mock.Mock, None, None]:
    with mock.patch.object(determined, "get_cluster_info") as mock_get_cluster_info:
        mock_info = mock.Mock()
        mock_info.trial = mock.Mock()
        mock_info.trial.hparams = {
            "output_dir": "dummy",
            "task_args": {
                "num_fewshot": 0,
                "task_name": "ARC",
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
            "output_dir": "dummy",
            "hf_upload_dir": "hf_dummy",
            "description": "det_description",
            "llm_args": {"model_name": "det_model"},
            "task_args": {
                "num_fewshot": 0,
                "num_samples": 10,
                "max_tokens": 100,
                "batch_size": 16,
                "task_subjects": ["subject1", "subject2"],
                "task_name": "ARC",
                "judge_model_args": {"model_name": "det_judge"},
            },
        }
        mock_get_cluster_info.return_value = mock_info
        yield mock_get_cluster_info


def test_determined_context_minimal(mock_get_cluster_info_minimal: mock.Mock) -> None:
    # Test that values from determined configuration are used if given but otherwise run.py values are used.
    # Here, we specify as few values as possible in the hparams to test that run.py values are used.
    with DeterminedContext(
        num_samples=10000,
        max_tokens=111,
        num_fewshot=555,  # overriden by hparams
        task_name="GSM8K",  # overriden by hparams
        task_subjects=None,
        output_dir=Path("dummyXXX"),  # overriden by hparams
        hf_upload_dir="dummy123",
        llm_args={"model_name": "run_model"},
        judge_model_args={"model_name": "run_judge"},
        batch_size=1,
        description="d",
    ) as ctx:
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.hparams is not None
        assert ctx._core_context is not None
        assert ctx.config.llm_class.__name__ == OpenAIModel.__name__
        assert ctx.config.num_samples == 10000
        assert ctx.config.max_tokens == 111
        assert ctx.config.num_fewshot == 0
        assert ctx.config.task_name == "ARC"
        assert ctx.config.task_subjects is None
        assert ctx.config.output_dir == Path("dummy")
        assert ctx.config.hf_upload_dir == "dummy123"
        assert ctx.config.llm_args == {"model_name": "run_model"}
        assert ctx.config.llm_judge_class is not None
        assert ctx.config.llm_judge_class.__name__ == OpenAIModel.__name__
        assert ctx.config.judge_model_args == {"model_name": "run_judge"}
        assert ctx.config.judge_model_args is not None
        assert ctx.config.batch_size == 1
        assert ctx.config.description == "d"
        assert ctx.config.repeats == 1
    mock_get_cluster_info_minimal.assert_called()


def test_determined_context_maximal(mock_get_cluster_info_maximal: mock.Mock) -> None:
    # Test that values from determined configuration are used if given but otherwise run.py values are used.
    # Here, we specify as many values as possible in the hparams to test that they are used.
    with DeterminedContext(
        num_samples=10000,  # overriden by hparams
        max_tokens=111,  # overriden by hparams
        num_fewshot=555,  # overriden by hparams
        task_name="GSM8K",  # overriden by hparams
        task_subjects=None,  # overriden by hparams
        output_dir=Path("dummyXXX"),  # overriden by hparams
        hf_upload_dir="dummy123",  # overriden by hparams
        llm_args={"model_name": "run_model"},  # overriden by hparams
        judge_model_args={"model_name": "run_judge"},  # overriden by hparams
        batch_size=1,  # overriden by hparams
        description="d",  # overriden by hparams
        repeats=100,
    ) as ctx:
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.hparams is not None
        assert ctx._core_context is not None
        assert ctx.config.llm_class.__name__ == OpenAIModel.__name__
        assert ctx.config.num_samples == 10
        assert ctx.config.max_tokens == 100
        assert ctx.config.num_fewshot == 0
        assert ctx.config.task_name == "ARC"
        assert ctx.config.task_subjects == ["subject1", "subject2"]
        assert ctx.config.output_dir == Path("dummy")
        assert ctx.config.hf_upload_dir == "hf_dummy"
        assert ctx.config.llm_args == {"model_name": "det_model"}
        assert ctx.config.llm_judge_class is not None
        assert ctx.config.llm_judge_class.__name__ == OpenAIModel.__name__
        assert ctx.config.judge_model_args == {"model_name": "det_judge"}
        assert ctx.config.batch_size == 16
        assert ctx.config.description == "det_description"
        assert ctx.config.repeats == 100
    mock_get_cluster_info_maximal.assert_called()


def test_local_context() -> None:
    with LocalContext(
        num_samples=10,
        num_fewshot=0,
        task_name="ARC",
        output_dir=Path("dummy"),
        hf_upload_dir="dummy22",
        llm_args={"model_name": "m"},
        judge_model_args={"model_name": "judge"},
        batch_size=1,
        repeats=1,
    ) as ctx:
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.config.llm_class.__name__ == OpenAIModel.__name__
        assert ctx.config.num_samples == 10
        assert ctx.config.num_fewshot == 0
        assert ctx.config.task_name == "ARC"
        assert ctx.config.output_dir == Path("dummy")
        assert ctx.config.hf_upload_dir == "dummy22"
        assert ctx.config.llm_args == {"model_name": "m"}
        assert ctx.config.llm_judge_class is not None
        assert ctx.config.llm_judge_class.__name__ == OpenAIModel.__name__
        assert ctx.config.judge_model_args is not None
        assert ctx.config.repeats == 1


def test_local_context_without_judge_model_args() -> None:
    with LocalContext(
        task_name="ARC", num_fewshot=0, batch_size=1, output_dir=Path("dummy"), llm_args={"model_name": "m"}
    ) as ctx:
        assert ctx.config is not None
        assert ctx.config.llm_judge_class is None


@pytest.mark.parametrize(
    ("llm_args", "judge_model_args", "match"),
    [
        ({}, {}, "llm_args must set model_name"),
        ({"model_name": "m"}, {"temperature": "0"}, "judge_model_args must set model_name"),
    ],
)
def test_local_context_requires_model_names(
    llm_args: dict[str, str], judge_model_args: dict[str, str], match: str
) -> None:
    context = LocalContext(task_name="ARC", llm_args=llm_args, judge_model_args=judge_model_args)
    with pytest.raises(ValueError, match=match):
        context.__enter__()
