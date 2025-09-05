import pytest

from eval_framework.tasks.benchmarks.infinitebench import (
    InfiniteBench_CodeRun,
    InfiniteBench_EnDia,
    InfiniteBench_EnQA,
    InfiniteBench_MathFind,
    InfiniteBench_RetrieveKV2,
    InfiniteBench_RetrieveNumber,
    InfiniteBench_RetrievePassKey1,
)
from tests.utils import DatasetPatcher


class Test_InfiniteBench_CodeRun:
    @pytest.fixture
    def task(self) -> InfiniteBench_CodeRun:
        return InfiniteBench_CodeRun(0, None, None)

    def test_InfiniteBench_CodeRun_postprocessing(self, task: InfiniteBench_CodeRun) -> None:
        assert (
            task.post_process_generated_completion(
                completion_text="Some reasoning tokens. The return value is: 42", sample=None
            )
            == "42"
        )


class Test_InfiniteBench_EnDia:
    @pytest.fixture
    def task(self) -> InfiniteBench_EnDia:
        with DatasetPatcher(InfiniteBench_EnDia, num_fewshot=0) as patched_task:
            return patched_task

    def test_InfiniteBench_EnDia_postprocessing(self, task: InfiniteBench_EnDia) -> None:
        assert task.post_process_generated_completion(completion_text="Ally Jones", sample=None) == "ally jones"


class Test_InfiniteBench_EnQA:
    @pytest.fixture
    def task(self) -> InfiniteBench_EnQA:
        with DatasetPatcher(InfiniteBench_EnQA, num_fewshot=0) as patched_task:
            return patched_task

    def test_InfiniteBench_EnQA_postprocessing(self, task: InfiniteBench_EnQA) -> None:
        assert (
            task.post_process_generated_completion(
                completion_text="Avalon's brother-in-law is Jayson Mac Andrew", sample=None
            )
            == "avalon's brother-in-law is jayson mac andrew"
        )


class Test_InfiniteBench_MathFind:
    @pytest.fixture
    def task(self) -> InfiniteBench_MathFind:
        with DatasetPatcher(InfiniteBench_MathFind, num_fewshot=0) as patched_task:
            return patched_task

    def test_InfiniteBench_MathFind_postprocessing(self, task: InfiniteBench_MathFind) -> None:
        assert task.post_process_generated_completion(completion_text="89", sample=None) == "89"
        assert task.post_process_generated_completion(completion_text="no numbers", sample=None) == "[invalid]"


class Test_InfiniteBench_RetrieveKV2:
    @pytest.fixture
    def task(self) -> InfiniteBench_RetrieveKV2:
        with DatasetPatcher(InfiniteBench_RetrieveKV2, num_fewshot=0) as patched_task:
            return patched_task

    def test_InfiniteBench_RetrieveKV2_postprocessing(self, task: InfiniteBench_RetrieveKV2) -> None:
        assert (
            task.post_process_generated_completion(completion_text="89018a30-9783-4b4c-99dc-25cf90100e58", sample=None)
            == "89018a30-9783-4b4c-99dc-25cf90100e58"
        )
        assert task.post_process_generated_completion(completion_text="wrong solution", sample=None) == "[invalid]"


class Test_InfiniteBench_RetrieveNumber:
    @pytest.fixture
    def task(self) -> InfiniteBench_RetrieveNumber:
        with DatasetPatcher(InfiniteBench_RetrieveNumber, num_fewshot=0) as patched_task:
            return patched_task

    def test_InfiniteBench_RetrieveNumber_postprocessing(self, task: InfiniteBench_RetrieveNumber) -> None:
        assert task.post_process_generated_completion(completion_text="7222999916", sample=None) == "7222999916"
        assert task.post_process_generated_completion(completion_text="no number provided", sample=None) == "[invalid]"


class Test_InfiniteBench_RetrievePassKey1:
    @pytest.fixture
    def task(self) -> InfiniteBench_RetrievePassKey1:
        with DatasetPatcher(InfiniteBench_RetrievePassKey1, num_fewshot=0) as patched_task:
            return patched_task

    def test_InfiniteBench_RetrievePassKey1_postprocessing(self, task: InfiniteBench_RetrievePassKey1) -> None:
        assert task.post_process_generated_completion(completion_text="03182", sample=None) == "03182"
        assert task.post_process_generated_completion(completion_text="no key provided", sample=None) == "[invalid]"
