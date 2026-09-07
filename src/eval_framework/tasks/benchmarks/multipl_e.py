"""MultiPL-E: translations of HumanEval and MBPP into 6 programming languages.

Corresponds to the following OLMES task suites:

  multipl_e_humaneval:6lang::olmo3:n32:v2 -> MultiPLEHumanEval, one SUBJECTS entry per language
    (cpp, java, js, php, rs, sh); pass task_subjects=["cpp"] etc. to run a single language.

  multipl_e_mbpp:6lang::olmo3:n32:v2 -> MultiPLEMBPP, same per-language subjects.

Recommended EvalConfig settings for full OLMES replication:
  repeats: 32
  llm_args: {sampling_params: {temperature: 0.6, top_p: 0.6}}
  max_tokens: 1024
  fewshot: 0

Paper: https://ieeexplore.ieee.org/abstract/document/10103177
"""

from typing import Any

from eval_framework.metrics.completion.multipl_e_assertion import MultiPLECodeAssertion, MultiPLEMetricContext
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE

MULTIPL_E_STOP_TOKENS: dict[str, list[str]] = {
    "cpp": ["\n}", "}\n//"],
    "java": ["\n    }\n", "}\n}", "}\n\n", "\n    public static void main", "\n    // Write"],
    "js": ["\nfunction ", "\n/*", "\n//", "\nconsole.log"],
    "php": ["\nfunction", "\n?>", "\n//", "\n#"],
    "rs": ["\n}"],
    "sh": ["}\n", "\n}"],
}


class _BaseMultiPLE(BaseTask[str]):
    """Abstract base for the MultiPL-E OLMES tasks, one per source dataset (HumanEval, MBPP).

    Each language is a SUBJECT rather than a separate class. Subclasses must define:
      - NAME (str): human-readable task name
      - MULTIPL_E_DATASET_PREFIX (str): HF dataset config prefix, "humaneval" or "mbpp"
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"  # no dedicated fewshot split; 0-shot is expected
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [MultiPLECodeAssertion]
    SUBJECTS = list(MULTIPL_E_STOP_TOKENS.keys())
    LANGUAGE = Language.ENG
    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    MULTIPL_E_DATASET_PREFIX: str  # "humaneval" or "mbpp", overridden by each dataset subclass

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, (
            "MultiPL-E does not support few-shot prompting (there are no gold examples for MultiPL-E)."
        )
        super().__init__(num_fewshot)
        # The runner resolves the model-level stop_sequences once per task instance, before any
        # subject/language is loaded, so this has to cover every language up front. The exact
        # per-language stop tokens are still applied in post_process_generated_completion below.
        all_stop_tokens = {token for tokens in MULTIPL_E_STOP_TOKENS.values() for token in tokens}
        self.stop_sequences: list[str] = sorted(all_stop_tokens)
        self.max_tokens: int = 1024

    def _load_dataset(self, subject: str) -> None:
        hf_dataset = self._load_hf_dataset(
            path=self.DATASET_PATH,
            name=f"{self.MULTIPL_E_DATASET_PREFIX}-{subject}",
        )
        self.dataset = self._shuffle_splits(hf_dataset)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # The prompt field already contains a complete function signature (and any leading
        # docstring / type annotations) in the target language. No additional formatting
        # is applied, matching the oe_eval behaviour (use_chat_format=False).
        return item["prompt"]

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        # Evaluation is entirely test-based; there is no single ground-truth string.
        return None

    def _get_context(self, item: dict[str, Any]) -> MultiPLEMetricContext:
        return MultiPLEMetricContext(
            prompt=item["prompt"],
            tests=item["tests"],
            language=item["language"],
        )


class MultiPLEHumanEval(_BaseMultiPLE):
    """MultiPL-E HumanEval — OLMES variant (nuprl/MultiPL-E, humaneval-*, test split).

    One SUBJECTS entry per language (cpp, java, js, php, rs, sh); corresponds to
    ``multipl_e_humaneval:6lang::olmo3:n32:v2`` in oe_eval, or its per-language
    ``multipl_e_humaneval:{cpp,java,js,php,rs,sh}::olmo3:n32:v2`` variants via task_subjects.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPL-E HumanEval OLMES"
    MULTIPL_E_DATASET_PREFIX = "humaneval"


class MultiPLEMBPP(_BaseMultiPLE):
    """MultiPL-E MBPP — OLMES variant (nuprl/MultiPL-E, mbpp-*, test split).

    One SUBJECTS entry per language (cpp, java, js, php, rs, sh); corresponds to
    ``multipl_e_mbpp:6lang::olmo3:n32:v2`` in oe_eval, or its per-language
    ``multipl_e_mbpp:{cpp,java,js,php,rs,sh}::olmo3:n32:v2`` variants via task_subjects.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPL-E MBPP OLMES"
    MULTIPL_E_DATASET_PREFIX = "mbpp"
