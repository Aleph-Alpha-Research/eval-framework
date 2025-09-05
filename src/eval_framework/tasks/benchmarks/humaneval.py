from typing import Any

from eval_framework.metrics.completion.code_assertion import CodeCompletionAssertion
from eval_framework.shared.types import BaseMetricContext
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, Sample

CODE_TO_EXECUTE = """
{start_of_code}
{completion_text}
{test_code}
try:
  check({entry_point})
  print(True)
except Exception as e:
  print(e)
  print(False)
"""


class HumanEvalMetricContext(BaseMetricContext):
    test: str
    entry_point: str
    prompt: str


class HumanEval(BaseTask[str]):
    """HumanEval dataset: https://huggingface.co/datasets/openai/openai_humaneval/"""

    NAME = "Human Eval"
    DATASET_PATH = "openai/openai_humaneval"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"  # (there is no dedicated split, few-shot is not expected for this dataset)
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [CodeCompletionAssertion]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)
        self.stop_sequences: list[str] = ["```"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"```python\n{item['prompt'].lstrip()}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return "Success"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return item["canonical_solution"]

    def _get_context(self, item: dict[str, Any]) -> HumanEvalMetricContext:
        return HumanEvalMetricContext(
            test=item["test"],
            entry_point=item["entry_point"],
            prompt=item["prompt"],
        )

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        assert sample is not None and sample.context is not None
        assert isinstance(sample.context, HumanEvalMetricContext), "Expected HumanEvalMetricContext"
        context = sample.context

        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]

        entry_point = context.entry_point
        test_code = context.test
        start_of_code = context.prompt
        formatted_code = CODE_TO_EXECUTE.format(
            start_of_code=start_of_code,
            completion_text=completion_text,
            test_code=test_code,
            entry_point=entry_point,
        )

        return formatted_code


class HumanEvalInstruct(HumanEval):
    # See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/humaneval/humaneval_instruct.yaml
    NAME = "Human Eval Instruct"
    CUE_PREFIX = "Here is the completed function:\n```python\n"

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for Human Eval Instruct"
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        instruction_text = (
            "Write a solution to the following problem and make sure that "
            f"it passes the tests:\n```python\n{item['prompt'].lstrip()}"
        )
        return instruction_text

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return self.CUE_PREFIX + item["prompt"].lstrip()
