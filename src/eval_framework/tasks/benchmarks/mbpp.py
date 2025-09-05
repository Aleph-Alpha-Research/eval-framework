import ast
import logging
import re
from typing import Any

from eval_framework.metrics.completion.code_assertion import (
    CodeCompletionAssertion,
)
from eval_framework.shared.types import BaseMetricContext
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample

logger = logging.getLogger(__name__)

BEGIN = "```python"
END = "```"


class MBPPMetricContext(BaseMetricContext):
    tests_code: str


class MBPP(BaseTask[str]):
    """
    MBPP provides both the problem statement and the test cases upfront. It says, "Here's the problem and here are the
    tests; write code that passes them.". Note that LLMs can cheat and only write code that passes the tests without
    solving the given problem.

    MBPP_PROMPT_WITHOUT_TESTS, on the other hand, only gives you the problem statement and function signature
    initially. It says, "Here's the problem and function signature; write code, then we'll run tests later."
    """

    NAME = "MBPP"
    DATASET_PATH = "google-research-datasets/mbpp"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [CodeCompletionAssertion]
    SUBJECTS = ["full"]  # , "sanitized"]  # these are HF dataset SUBSETS!
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)

        self.stop_sequences = [END]

    @staticmethod
    def _code_expander(code: str, gt_asserts: str) -> str:
        """
        code variable carries the LLM-generated code snippet. We append the asserts for code testing
        here. If no valid code is found in the LLM output, this function is not called.
        Important: gt_asserts come as a stringiied list of assert strings. We safely reconvert this string
        back to the list of of individual assert statements (also strings) by ast.literal_eval
        """
        if not gt_asserts:  # no ground truth (data asserts) are given, we return the original code
            return code
        gt_asserts = ast.literal_eval(gt_asserts)  # never use eval!
        if not isinstance(gt_asserts, list):
            logger.info("*** WARNING, we expect a list of ground truth asserts here! Sample can not be finalized")
            return code
        postfix = ""
        stacked_asserts = ""
        for gt_assert in gt_asserts:
            stacked_asserts += "    " + gt_assert + "\n"
        postfix = "try:\n" + stacked_asserts + "    score = True\nexcept:\n    score = False\nprint(score)"
        return code + postfix

    @staticmethod
    def _get_function_name(line: str) -> str:
        match = re.search(r"def\s+(\w+)\s*\(", line)
        function_name = ""
        if match:
            function_name = match.group(1)
        return function_name

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """
        Passing selected task and tests depending on zero or few-shot setting
        """
        tests = "\n".join(item["test_list"])
        text = item["text"] if "text" in item else item["prompt"]

        instruction_text = f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{tests}\n"  # noqa E501
        return instruction_text

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return BEGIN

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        """
        asserts are being passed as ground_truth, this is expected by CodeCompletionAssertion metrics
        """
        return f"{item['test_list']}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = item["code"]
        assert target is not None
        assert isinstance(target, str)
        return f"{BEGIN}\n" + target + f"\n{END}"

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        fewshot_examples = self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot)
        return fewshot_examples

    def _get_context(self, item: dict[str, Any]) -> MBPPMetricContext:
        return MBPPMetricContext(tests_code="\n".join(item["test_list"]))

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        assert sample is not None

        if BEGIN in completion_text:
            completion_text = completion_text.split(f"{BEGIN}\n")[1]

        if END in completion_text:
            completion_text = completion_text.split(END)[0]

        extracted_code = completion_text + "\n"
        mbpp_ground_truth = str(sample.ground_truth)
        code = self._code_expander(extracted_code, mbpp_ground_truth)
        return code


class MBPP_SANITIZED(MBPP):
    NAME = "MBPP_SANITZED"
    SUBJECTS = ["sanitized"]


class MBPP_PROMPT_WITHOUT_TESTS(MBPP):
    """
    MBPP provides both the problem statement and the test cases upfront. It says, "Here's the problem and here are the
    tests; write code that passes them.". Note that LLMs can cheat and only write code that passes the tests without
    solving the given problem.

    MBPP_PROMPT_WITHOUT_TESTS, on the other hand, only gives you the problem statement and function signature
    initially. It says, "Here's the problem and function signature; write code, then we'll run tests later."
    """

    NAME = "MBPP_PROMPT_WITHOUT_TESTS"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """
        Passing selected task and tests depending on zero or few-shot setting
        """
        text = item["text"] if "text" in item else item["prompt"]
        instruction_text = f"You are an expert Python programmer, and here is your task: {text}\n\n"  # noqa E501
        return instruction_text

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        function_header = self._get_function_header(item["code"])
        return f"{BEGIN}\n{function_header}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = item["code"]
        assert target is not None
        assert isinstance(target, str)
        return f"{BEGIN}\n" + target + f"\n{END}"

    @staticmethod
    def _get_function_header(line: str) -> str:
        match = re.search(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*?\)\s*:", line, re.MULTILINE)
        postfix = ""
        if match is not None:  # extract up to next open parenthesis in the found substring
            postfix = line[match.start() :]
            match = re.search(r"\)", postfix)
            if match is not None:
                end = match.start()
                postfix = postfix[: end + 1]
            else:
                postfix = ""

        if postfix == "":
            return postfix
        return f"{postfix.strip()}:"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        assert sample is not None

        if BEGIN in completion_text:
            completion_text = completion_text.split(BEGIN)[1]

        if END in completion_text:
            completion_text = completion_text.split(END)[0]

        extracted_code = completion_text + "\n"
        mbpp_ground_truth = str(sample.ground_truth)
        function_header = self._get_function_header(sample.messages[-1].content)
        code = self._code_expander(extracted_code, mbpp_ground_truth)
        return function_header + code


class MBPP_PROMPT_WITHOUT_TESTS_SANITIZED(MBPP_PROMPT_WITHOUT_TESTS):
    NAME = "MBPP_PROMPT_WITHOUT_TESTS_SANITIZED"
    SUBJECTS = ["sanitized"]
