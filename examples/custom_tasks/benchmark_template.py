"""
Benchmark Template for Eval Framework

This template provides a starting point for creating new benchmarks.
Copy this file and fill in the TODO sections with your specific implementation.

Usage:
1. Copy this file to your project
2. Replace "YourBenchmark" with your actual benchmark name
3. Fill in all TODO sections
4. Register your task in task_names.py
5. Add tests in tests/tasks/

Example: Geography Question Answering benchmark that tests knowledge of world capitals.
"""

from typing import Any

from eval_framework.metrics.completion_metrics import AccuracyCompletion  # Import your metrics
from eval_framework.tasks.base import BaseTask, ResponseType, Sample
from eval_framework.tasks.dataloader import Dataloader


class YourBenchmarkTask(BaseTask[str]):  # Replace with your class name
    # === REQUIRED CONFIGURATION ===
    NAME = "YourBenchmark"  # Set display name
    DATASET_PATH = "your_org/your_dataset"  # Set HuggingFace dataset path
    SAMPLE_SPLIT = "test"  # Set split for evaluation samples
    FEWSHOT_SPLIT = "train"  # Set split for few-shot examples
    RESPONSE_TYPE = ResponseType.COMPLETION  # Choose COMPLETION or LOGLIKELIHOODS
    METRICS = [AccuracyCompletion]  # List your metric classes
    SUBJECTS = ["subject1", "subject2"]  # Define your subjects/categories

    def __init__(self, dataloader: Dataloader, subjects: list[str] | None = None, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.subjects = subjects or self.SUBJECTS

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Generate the instruction/question text for a sample."""
        # Format your question from the dataset item
        # Example: return f"Q: What's the capital of {item['country']}?"
        raise NotImplementedError("Implement your question formatting")

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Extract the correct answer from a dataset item."""
        # Extract the correct answer from your dataset
        # Example: return f"A: {item['capital']}."
        raise NotImplementedError("Implement your answer extraction")

    # === OPTIONAL METHODS TO CUSTOMIZE ===

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
        """System message content (optional)."""
        # Add system prompt if needed
        # Example: return "Answer the geography questions accurately."
        return None

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        """Text to append as assistant cue (optional)."""
        # Add cue text if needed (e.g., "Answer:", "A:")
        # Example: return "A:"
        return ""

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        """For loglikelihood tasks: list of answer choices (required for LOGLIKELIHOODS)."""
        # Return list of choices for multiple choice tasks
        # Example: return [item['choice_a'], item['choice_b'], item['choice_c'], item['choice_d']]
        return None

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """Post-process model completions (optional)."""
        # Add any post-processing logic (e.g., extract final answer)
        # Example: return completion_text.split("Answer:")[-1].strip()
        return completion_text

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        """Custom few-shot sampling logic (optional)."""
        # Implement custom sampling if needed, otherwise use default
        return self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot)


# === EXAMPLE IMPLEMENTATION ===
# Here's a example showing how the template would be filled in:


class GeographyQATask(BaseTask[str]):
    """Example implementation: Geography Question Answering benchmark."""

    # Required configuration
    NAME = "GeographyQA"
    DATASET_PATH = "example/geography_qa"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["Europe", "Asia"]

    def __init__(self, dataloader: Dataloader, subjects: list[str] | None = None, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.subjects = subjects or self.SUBJECTS

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Format the question from the dataset item."""
        return f"Q: What's the capital of {item['country']}?"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Extract the correct answer from the dataset item."""
        return f"A: {item['capital']}."

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
        """Provide context about the task."""
        return "Answer the geography questions accurately."

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        """Start the model's response with 'A:'."""
        return "A:"
