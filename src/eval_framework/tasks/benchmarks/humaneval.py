from typing import Any

from eval_framework.metrics.completion.code_assertion import CodeCompletionAssertion
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.shared.types import BaseMetricContext
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.task_style import BPBStyle

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

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
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


class HumanEvalBPB(HumanEval):
    """
    HumanEval variant that scores loglikelihood of the gold canonical solution.
    Reports bits-per-byte on the reference completion.
    """

    NAME = "Human Eval BPB"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [BitsPerByteLoglikelihood]

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return " " + item["canonical_solution"]

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        gt = self._get_ground_truth(item)
        return [gt] if gt else None


class HumanEval_OLMES(HumanEval):
    """HumanEval OLMES variant replicating codex_humaneval:3shot::olmo3:n32:v2 from oe_eval.

    Recommended EvalConfig settings for full replication:
        repeats: 32
        llm_args: {sampling_params: {temperature: 0.6, top_p: 0.6}}
    """

    NAME = "Human Eval OLMES"

    def __init__(self, num_fewshot: int = 3) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["\nclass", "\nif", "\nprint", "\n#", "\n```", "\n```\n\n", "<|eot_id|>"]
        self.max_tokens = 1024

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return "```python\n" + item["prompt"]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return item["canonical_solution"] + "```"


class HumanEvalInstruct(HumanEval):
    # See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/humaneval/humaneval_instruct.yaml
    NAME = "Human Eval Instruct"
    CUE_PREFIX = "Here is the completed function:\n```python\n"

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for Human Eval Instruct"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        instruction_text = (
            "Write a solution to the following problem and make sure that "
            f"it passes the tests:\n```python\n{item['prompt'].lstrip()}"
        )
        return instruction_text

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return self.CUE_PREFIX + item["prompt"].lstrip()


# fmt: off
# Fixed 3-shot fewshot examples for codex_humaneval_gold_bpb_3shot.
# Source: HumanEval test split, task_ids HumanEval/112, HumanEval/29, HumanEval/1 (in that order).
_CODEX_HUMANEVAL_FEWSHOTS: list[dict[str, Any]] = [
    {
        "task_id": "HumanEval/112",
        "entry_point": "reverse_delete",
        # The HumanEval/112 prompt starts with "\n" in the dataset.  In the
        # reference (olmo_eval) this becomes the very first character of the
        # pre-baked ctx string, and ConcatFormatter strips it when formatting a
        # single-message context.  We strip it here so that our multi-message
        # context produces the same output.
        "prompt": 'def reverse_delete(s,c):\n    """Task\n    We are given two strings s and c, you have to deleted all the characters in s that are equal to any character in c\n    then check if the result string is palindrome.\n    A string is called palindrome if it reads the same backward as forward.\n    You should return a tuple containing the result string and True/False for the check.\n    Example\n    For s = "abcde", c = "ae", the result should be (\'bcd\',False)\n    For s = "abcdef", c = "b"  the result should be (\'acdef\',False)\n    For s = "abcdedcba", c = "ab", the result should be (\'cdedc\',True)\n    """\n',#noqa
        "canonical_solution": "    s = ''.join([char for char in s if char not in c])\n    return (s,s[::-1] == s)\n",
    },
    {
        "task_id": "HumanEval/29",
        "entry_point": "filter_by_prefix",
        "prompt": "from typing import List\n\n\ndef filter_by_prefix(strings: List[str], prefix: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that start with a given prefix.\n    >>> filter_by_prefix([], 'a')\n    []\n    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')\n    ['abc', 'array']\n    \"\"\"\n",#noqa
        "canonical_solution": "    return [x for x in strings if x.startswith(prefix)]\n",
    },
    {
        "task_id": "HumanEval/1",
        "entry_point": "separate_paren_groups",
        "prompt": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",#noqa
        "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string.clear()\n\n    return result\n",#noqa
    },
]

# Replacement fewshot used when the test item coincides with one of the standard
# three (to avoid self-referential fewshots, mirroring the olmo_eval behaviour).
_STRLEN_FEWSHOT: dict[str, Any] = {
    "task_id": "HumanEval/23",
    "entry_point": "strlen",
    "prompt": '\n\ndef strlen(string: str) -> int:\n    """ Return length of given string\n    >>> strlen(\'\')\n    0\n    >>> strlen(\'abc\')\n    3\n    """\n',#noqa
    "canonical_solution": "    return len(string)\n",
}
# fmt: on

_STANDARD_FEWSHOT_IDS: frozenset[str] = frozenset(d["task_id"] for d in _CODEX_HUMANEVAL_FEWSHOTS)


class _CodexHumanEval_Base(BaseTask[str]):
    """Shared base for codex_humaneval_gold_bpb_3shot-compatible HumanEval variants.

    Follows the TASK_STYLER pattern (like ARC):
    - ``_get_raw_question`` → ``item["prompt"]`` (function signature + docstring)
    - ``_get_choices``      → ``[item["canonical_solution"]]``
    - ``_get_correct_index`` → ``0``

    ``RESPONSE_TYPE`` and ``METRICS`` are provided by the ``TASK_STYLER``.

    BPBStyle normally prepends ``" "`` to the scored completion, but HumanEval
    prompts already end with ``"\\n"`` which ConcatFormatter strips from the last
    USER message.  ``_get_possible_completions`` is therefore overridden to omit
    that space so the completion starts directly with the four-space indent of
    the function body, matching the olmo_eval reference.  The fewshot *target*
    retains the leading space via ``BPBStyle.get_fewshot_target_text`` because
    those messages are not the final USER turn (no stripping).
    """

    DATASET_PATH = "openai/openai_humaneval"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["prompt"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return [item["canonical_solution"]]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return 0

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        # Skip BPBStyle's default " " prefix — the prompt's trailing "\n" is
        # stripped by ConcatFormatter, so no extra space is needed.
        return [item["canonical_solution"]]

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        test_id = item.get("task_id", "")
        if test_id in _STANDARD_FEWSHOT_IDS:
            # Avoid self-referential fewshots: drop the test item's own example
            # and substitute strlen (HumanEval/23), mirroring olmo_eval.
            base = [d for d in _CODEX_HUMANEVAL_FEWSHOTS if d["task_id"] != test_id]
            return (base + [_STRLEN_FEWSHOT])[: self.num_fewshot]
        return _CODEX_HUMANEVAL_FEWSHOTS[: self.num_fewshot]


class CodexHumanEval_BPB(_CodexHumanEval_Base):
    """BPB-only HumanEval that matches codex_humaneval_gold_bpb_3shot.

    Prompt: ``{prompt}`` (function signature + docstring, verbatim)
    Scored completion: ``{canonical_solution}``
    """

    NAME = "CodexHumanEval_BPB"
    TASK_STYLER = BPBStyle(question_prefix="", cue_text="", trailing_newline=False)
