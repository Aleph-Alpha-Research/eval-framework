import re

import pytest

from eval_framework.tasks.benchmarks.gpqa import GPQA, GPQA_COT
from tests.utils import DatasetPatcher


class TestGPQA:
    @pytest.fixture
    def gpqa_task(self) -> GPQA:
        with DatasetPatcher(GPQA, num_fewshot=1) as patched_task:
            return patched_task

    def test_ground_truth_in_completion(self, gpqa_task: GPQA) -> None:
        subject = gpqa_task.SUBJECTS[0]
        gpqa_task._load_dataset(subject)
        for x in range(5):
            item = gpqa_task.dataset[gpqa_task.SAMPLE_SPLIT][x]
            ground_truths = set()
            for _ in range(6):
                possible_completions = gpqa_task._get_possible_completions(item)
                ground_truth = gpqa_task._get_ground_truth(item.copy())
                ground_truths.add(ground_truth)
                assert possible_completions is not None
                assert ground_truth in possible_completions
                assert possible_completions is not None
                assert ground_truth in possible_completions
            assert len(ground_truths) == 1  # check that we are not random from the ordering in one item


class TestGPQA_COT:
    @pytest.fixture
    def gpqa_cot_task(self) -> GPQA_COT:
        with DatasetPatcher(GPQA_COT) as patched_task:
            return patched_task

    def test_get_possible_completions_marked_deterministic(self, gpqa_cot_task: GPQA_COT) -> None:
        subject = gpqa_cot_task.SUBJECTS[0]
        gpqa_cot_task._load_dataset(subject)
        item = gpqa_cot_task.dataset[gpqa_cot_task.SAMPLE_SPLIT][0]
        first_run = gpqa_cot_task._get_possible_completions_marked(item)
        second_run = gpqa_cot_task._get_possible_completions_marked(item)
        assert first_run == second_run  # deterministic output
        choices, correct_index = first_run
        assert len(choices) == 4
        assert 0 <= correct_index <= 3
        assert re.sub(r"\([A-Z]\)\s+", "", choices[correct_index]) == item["Correct Answer"]

    def test_get_ground_truth_returns_correct_letter(self, gpqa_cot_task: GPQA_COT) -> None:
        subject = gpqa_cot_task.SUBJECTS[0]
        gpqa_cot_task._load_dataset(subject)
        item = gpqa_cot_task.dataset[gpqa_cot_task.SAMPLE_SPLIT][0]
        choices, correct_index = gpqa_cot_task._get_possible_completions_marked(item)
        expected_correct_letter = choices[correct_index][1]
        result = gpqa_cot_task._get_ground_truth(item)
        assert result == expected_correct_letter

    def test_ground_truth_in_completion(self, gpqa_cot_task: GPQA_COT) -> None:
        subject = gpqa_cot_task.SUBJECTS[0]
        gpqa_cot_task._load_dataset(subject)
        item = gpqa_cot_task.dataset[gpqa_cot_task.SAMPLE_SPLIT][0]
        choices, _ = gpqa_cot_task._get_possible_completions_marked(item)
        ground_truth = gpqa_cot_task._get_ground_truth(item.copy())
        possible_completions = [f"({choice[1]})" for choice in choices]
        assert f"({ground_truth})" in possible_completions

    def test_ground_truth_in_completion_cot(self, gpqa_cot_task: GPQA_COT) -> None:
        subject = gpqa_cot_task.SUBJECTS[0]
        gpqa_cot_task._load_dataset(subject)
        for x in range(5):
            item = gpqa_cot_task.dataset[gpqa_cot_task.SAMPLE_SPLIT][x]
            ground_truths: set[str] = set()
            for _ in range(6):
                choices, _ = gpqa_cot_task._get_possible_completions_marked(item)
                ground_truth = gpqa_cot_task._get_ground_truth(item.copy())
                assert ground_truth is not None
                ground_truths.add(ground_truth)
                possible_completions = [f"({choice[1]})" for choice in choices]
                assert f"({ground_truth})" in possible_completions
            assert len(ground_truths) == 1
