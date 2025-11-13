import pytest

from eval_framework.metrics.completion.struct_eval_metrics import StructMetricContext
from eval_framework.tasks.benchmarks.struct_eval import StructEval


class TestStructEval:
    @pytest.fixture
    def struct_eval_task(self) -> StructEval:
        return StructEval()

    @pytest.mark.parametrize("subject", StructEval.SUBJECTS)
    def test_struct_eval_task_loads_dataset_for_subjects(self, struct_eval_task: StructEval, subject: str) -> None:
        assert struct_eval_task.DATASET_PATH == "TIGER-Lab/StructEval"
        struct_eval_task._load_dataset(subject)
        assert struct_eval_task.dataset is not None
        assert len(struct_eval_task.dataset) > 0
        assert all(item["task_name"] == subject for item in struct_eval_task.dataset[struct_eval_task.SAMPLE_SPLIT]), (
            f"Dataset for subject {subject} does not match expected task name."
        )

    def test_struct_eval_task_has_eval_kwargs(self, struct_eval_task: StructEval) -> None:
        assert len(struct_eval_task.SUBJECTS) > 0 and isinstance(struct_eval_task.SUBJECTS[0], str)
        struct_eval_task._load_dataset(struct_eval_task.SUBJECTS[0])
        assert struct_eval_task.dataset is not None, "Dataset should not be None after loading."
        eval_context = struct_eval_task._get_context(struct_eval_task.dataset[struct_eval_task.SAMPLE_SPLIT][0])
        assert eval_context is not None, "Eval context should not be None."
        assert isinstance(eval_context, StructMetricContext), "Eval context should be a StructMetricContext."
        assert isinstance(eval_context.output_type, str) and len(eval_context.output_type) > 0, (
            "Eval context output_type should be a non-empty string."
        )
        assert isinstance(eval_context.paths, list) and len(eval_context.paths) > 0, (
            "Eval context paths should be a non-empty list."
        )

    @pytest.mark.parametrize(
        "sample_text",
        [
            "bla   . ```python\nsample\ncode\n``` bla",
            "bla   . ```python\nsample\ncode``` bla",
            "```html\nsample\ncode```",
            "sample\ncode",
        ],
    )
    def test_post_processing_removes_fences(self, struct_eval_task: StructEval, sample_text: str) -> None:
        cleaned_sample = struct_eval_task.post_process_generated_completion(sample_text)
        assert cleaned_sample == "sample\ncode", "Post-processing did not remove code fences correctly."
