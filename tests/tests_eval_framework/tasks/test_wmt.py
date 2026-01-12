"""
Tests for WMT benchmark tasks using HuggingFace datasets.

Validates that WMT tasks load data deterministically from HuggingFace datasets
instead of sacrebleu file-based loading.
"""

import pytest

from eval_framework.tasks.benchmarks.wmt import (
    WMT14,
    WMT14_INSTRUCT,
    WMT16,
    WMT16_INSTRUCT,
    WMT20,
    WMT20_INSTRUCT,
)


class TestWMTDatasetStructure:
    """Test that WMT tasks load data with correct structure from HuggingFace."""

    @pytest.mark.parametrize(
        "task_cls,subject",
        [
            (WMT14, "fr-en"),
            (WMT14, "en-fr"),
            (WMT16, "de-en"),
            (WMT16, "en-de"),
            (WMT20, "de-en"),
            (WMT20, "de-fr"),
        ],
    )
    def test_load_dataset_structure(self, task_cls: type, subject: str) -> None:
        """Test that dataset loads with correct structure."""
        task = task_cls(num_fewshot=0)
        task._load_dataset(subject)

        assert "test" in task.dataset
        assert len(task.dataset["test"]) > 0

        # Check item structure
        item = task.dataset["test"][0]
        assert "source" in item
        assert "target" in item
        assert "subject" in item
        assert item["subject"] == subject
        assert isinstance(item["source"], str)
        assert isinstance(item["target"], str)
        assert len(item["source"]) > 0
        assert len(item["target"]) > 0

    @pytest.mark.parametrize("task_cls", [WMT14, WMT16, WMT20])
    def test_deterministic_loading(self, task_cls: type) -> None:
        """Test that loading is deterministic across multiple runs."""
        subject = task_cls.SUBJECTS[0]

        # Load twice
        task1 = task_cls(num_fewshot=0)
        task1._load_dataset(subject)

        task2 = task_cls(num_fewshot=0)
        task2._load_dataset(subject)

        # Verify identical ordering after shuffle
        assert len(task1.dataset["test"]) == len(task2.dataset["test"])
        for i in range(min(10, len(task1.dataset["test"]))):
            assert task1.dataset["test"][i]["source"] == task2.dataset["test"][i]["source"]
            assert task1.dataset["test"][i]["target"] == task2.dataset["test"][i]["target"]


class TestWMTSampleGeneration:
    """Test WMT sample generation."""

    @pytest.mark.parametrize("task_cls", [WMT14, WMT16, WMT20])
    def test_sample_generation(self, task_cls: type) -> None:
        """Test that samples can be generated correctly."""
        task = task_cls(num_fewshot=0)
        samples = list(task.iterate_samples(num_samples=3))

        assert len(samples) == 3
        for sample in samples:
            assert sample.messages is not None
            assert len(sample.messages) > 0
            assert sample.ground_truth is not None

    @pytest.mark.parametrize("task_cls", [WMT14, WMT16, WMT20])
    def test_sample_with_fewshot(self, task_cls: type) -> None:
        """Test that few-shot samples are generated correctly."""
        task = task_cls(num_fewshot=1)
        samples = list(task.iterate_samples(num_samples=2))

        assert len(samples) == 2
        for sample in samples:
            # With fewshot, we should have more messages
            assert len(sample.messages) >= 2


class TestWMTInstructVariants:
    """Test WMT instruct variants."""

    @pytest.mark.parametrize(
        "task_cls,subject",
        [
            (WMT14_INSTRUCT, "fr-en"),
            (WMT16_INSTRUCT, "de-en"),
            (WMT20_INSTRUCT, "de-en"),
        ],
    )
    def test_instruct_sample_generation(self, task_cls: type, subject: str) -> None:
        """Test that instruct variants generate samples correctly."""
        task = task_cls(num_fewshot=0)
        samples = list(task.iterate_samples(num_samples=2))

        assert len(samples) == 2
        for sample in samples:
            assert sample.messages is not None
            # Check that the instruction format contains "translate"
            first_message_content = sample.messages[0].content
            assert "translate" in first_message_content.lower()


class TestWMTPostProcessing:
    """Test WMT post-processing methods."""

    def test_post_process_with_stop_sequence(self) -> None:
        """Test that stop sequences are handled correctly."""
        task = WMT16(num_fewshot=0)

        # Test various stop sequences
        text_with_stop = "Hello world.\nThis should be cut"
        result = task.post_process_generated_completion(text_with_stop)
        assert result == "Hello world"

        text_with_phrase = "Hello world phrase: extra"
        result = task.post_process_generated_completion(text_with_phrase)
        assert result == "Hello world"

    def test_instruct_post_process(self) -> None:
        """Test instruct variant post-processing."""
        task = WMT16_INSTRUCT(num_fewshot=0)

        # Test prefix removal
        text_with_prefix = "This is the translation: Hello world"
        result = task.post_process_generated_completion(text_with_prefix)
        assert result == "Hello world"
