import random
import re
from typing import Any

from eval_framework.metrics.completion.code_execution_pass_at_one import (
    CodeExecutionPassAtOne,
    CodeExecutionPassAtOneContext,
)
from eval_framework.tasks.base import (
    RANDOM_SEED,
    BaseTask,
    Language,
    ResponseType,
    Sample,
    SubjectType,
)
from eval_framework.tasks.utils import BIG_CODE_BENCH_PACKAGE_MAPPING, CallableSerializer, _parse_unittest_output

PROMPT_INSTRUCTION = (
    "Please provide a self-contained Python script, without tests or example usage, that solves the following "
    "problem in a markdown code block:\n"
)  # from https://arxiv.org/pdf/2406.15877 - Figure 14


RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem and passes "
    "corresponding tests:\n"
)  # from https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/generate.py#L149


class BigCodeBench(BaseTask[str]):
    """BigCodeBench dataset: https://huggingface.co/datasets/bigcode/bigcodebench"""

    NAME = "BigCodeBench"
    DATASET_PATH = "bigcode/bigcodebench"
    SAMPLE_SPLIT = "v0.1.4"
    FEWSHOT_SPLIT = "v0.1.4"  # (there is no dedicated split, few-shot is not expected for this dataset)
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [CodeExecutionPassAtOne]
    SUBJECTS = ["original", "calibrated"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for BigCodeBench"
        self.serializer = CallableSerializer()
        super().__init__(num_fewshot)

    def _load_dataset(self, subject: SubjectType) -> None:
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=None)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return PROMPT_INSTRUCTION + item["complete_prompt"]

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return RESPONSE_PREFIX + (item["code_prompt"] if item["subject"] == "calibrated" else "")

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item["canonical_solution"]  # Not needed for evaluation, as it is test based given the generated code

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return None

    @staticmethod
    def merge_snippets(code: str, test_code: str) -> str:
        # Add unittest.main() if not present (note that without "if" sometimes it just reports
        # "Ran 0 tests" errorneously).
        if "unittest.main(" not in test_code:
            test_code += "\n\nif __name__ == '__main__':\n  unittest.main()"

        # Combine the implementation code and test code
        combined_code = code + "\n\n" + test_code
        return combined_code

    def _get_context(self, item: dict[str, Any]) -> CodeExecutionPassAtOneContext:
        return CodeExecutionPassAtOneContext(
            run_env="python:3.12",  # os.environ.get("DOCKER_CODE_EXECUTION"),
            code_prompt=item["code_prompt"],
            test_code=item["test"],
            snippet_merge_fn=self.serializer.encode(self.merge_snippets),
            output_parse_fn=self.serializer.encode(_parse_unittest_output),
            package_downloads=BIG_CODE_BENCH_PACKAGE_MAPPING,
        )

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        if sample is not None and sample.context is not None and sample.subject == "calibrated":
            assert isinstance(sample.context, CodeExecutionPassAtOneContext), "Expected CodeExecutionPassAtOneContext"
            processed_text = (sample.context.code_prompt if sample.context is not None else "") + completion_text
        else:
            processed_text = extract_executable_code(completion_text)

        return processed_text


class BigCodeBenchInstruct(BigCodeBench):
    """BigCodeBench dataset: https://huggingface.co/datasets/bigcode/bigcodebench"""

    NAME = "BigCodeBenchInstruct"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return PROMPT_INSTRUCTION + item["instruct_prompt"]


class BigCodeBenchHard(BigCodeBench):
    """BigCodeBench dataset: https://huggingface.co/datasets/bigcode/bigcodebench-hard"""

    NAME = "BigCodeBenchHard"
    DATASET_PATH = "bigcode/bigcodebench-hard"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return PROMPT_INSTRUCTION + item["complete_prompt"]


class BigCodeBenchHardInstruct(BigCodeBenchHard):
    """BigCodeBench dataset: https://huggingface.co/datasets/bigcode/bigcodebench-hard"""

    NAME = "BigCodeBenchHardInstruct"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return PROMPT_INSTRUCTION + item["instruct_prompt"]


def extract_executable_code(llm_response: str) -> str:
    # Look for nested markdown+python pattern
    nested_pattern = r"```markdown.*?```python\s*(.*?)\s*```"
    nested_matches = re.findall(nested_pattern, llm_response, re.DOTALL)
    if nested_matches:
        return nested_matches[0].strip()

    # Look for python code blocks
    python_pattern = r"```python\s*(.*?)\s*```"
    python_matches = re.findall(python_pattern, llm_response, re.DOTALL)
    if python_matches:
        return python_matches[0].strip()

    # Look for markdown-only code blocks
    markdown_pattern = r"```markdown\s*(.*?)\s*```"
    markdown_matches = re.findall(markdown_pattern, llm_response, re.DOTALL)
    if markdown_matches:
        return markdown_matches[0].strip()

    # Look for generic code blocks as fallback
    generic_pattern = r"```\s*(.*?)\s*```"
    generic_matches = re.findall(generic_pattern, llm_response, re.DOTALL)
    if generic_matches:
        return generic_matches[0].strip()

    # If no code blocks found, return original response
    return llm_response
