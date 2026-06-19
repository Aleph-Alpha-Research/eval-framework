"""MultiPL-E: translations of HumanEval and MBPP into 6 programming languages.

Corresponds to the following OLMES task suites:

  multipl_e_humaneval:6lang::olmo3:n32:v2 (one task per language):
    multipl_e_humaneval:{cpp,java,js,php,rs,sh}::olmo3:n32:v2

  multipl_e_mbpp:6lang::olmo3:n32:v2 (one task per language):
    multipl_e_mbpp:{cpp,java,js,php,rs,sh}::olmo3:n32:v2

Recommended EvalConfig settings for full OLMES replication:
  repeats: 32
  llm_args: {sampling_params: {temperature: 0.6, top_p: 0.6}}
  max_tokens: 1024
  fewshot: 0

Paper: https://ieeexplore.ieee.org/abstract/document/10103177
"""

from typing import Any

from eval_framework.metrics.completion.multipl_e_assertion import MultiPLECodeAssertion, MultiPLEMetricContext
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, Sample

MULTIPL_E_STOP_TOKENS: dict[str, list[str]] = {
    "cpp": ["\n}", "}\n//"],
    "java": ["\n    }\n", "}\n}", "}\n\n", "\n    public static void main", "\n    // Write"],
    "js": ["\nfunction ", "\n/*", "\n//", "\nconsole.log"],
    "php": ["\nfunction", "\n?>", "\n//", "\n#"],
    "rs": ["\n}"],
    "sh": ["}\n", "\n}"],
}


class _MPLE_Base(BaseTask[str]):
    """Abstract base for all MultiPL-E OLMES per-language tasks.

    Subclasses must define:
      - NAME (str): human-readable task name
      - MULTIPL_E_LANGUAGE (str): language code used in the HF dataset config
        and stop-token lookup (e.g. "cpp", "java", "js", "php", "rs", "sh")
      - MULTIPL_E_DATASET_PREFIX (str): HF dataset config prefix,
        either "humaneval" or "mbpp"
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"  # no dedicated fewshot split; 0-shot is expected
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [MultiPLECodeAssertion]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    MULTIPL_E_LANGUAGE: str  # overridden by each language subclass
    MULTIPL_E_DATASET_PREFIX: str  # "humaneval" or "mbpp"

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, (
            "MultiPL-E does not support few-shot prompting (there are no gold examples for MultiPL-E)."
        )
        super().__init__(num_fewshot)
        self.stop_sequences: list[str] = MULTIPL_E_STOP_TOKENS[self.MULTIPL_E_LANGUAGE]
        self.max_tokens: int = 1024

    def _load_dataset(self, subject: str) -> None:
        hf_dataset = self._load_hf_dataset(
            path=self.DATASET_PATH,
            name=f"{self.MULTIPL_E_DATASET_PREFIX}-{self.MULTIPL_E_LANGUAGE}",
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

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """Apply language-specific stop sequences to trim the model's raw continuation."""
        for stop_seq in self.stop_sequences:
            if stop_seq in completion_text:
                completion_text = completion_text.split(stop_seq)[0]
        return completion_text


class _MPLE_HumanEval_Base(_MPLE_Base):
    MULTIPL_E_DATASET_PREFIX = "humaneval"


class MultiPLE_NUPRL_EN_HumanEval_Cpp(_MPLE_HumanEval_Base):
    """MultiPL-E HumanEval in C++ — OLMES variant (nuprl/MultiPL-E, humaneval-cpp, test split).

    Corresponds to ``multipl_e_humaneval:cpp::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_HumanEval_Cpp"
    MULTIPL_E_LANGUAGE = "cpp"


class MultiPLE_NUPRL_EN_HumanEval_Java(_MPLE_HumanEval_Base):
    """MultiPL-E HumanEval in Java — OLMES variant (nuprl/MultiPL-E, humaneval-java, test split).

    Corresponds to ``multipl_e_humaneval:java::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_HumanEval_Java"
    MULTIPL_E_LANGUAGE = "java"


class MultiPLE_NUPRL_EN_HumanEval_JS(_MPLE_HumanEval_Base):
    """MultiPL-E HumanEval in JavaScript — OLMES variant (nuprl/MultiPL-E, humaneval-js, test split).

    Corresponds to ``multipl_e_humaneval:js::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_HumanEval_JS"
    MULTIPL_E_LANGUAGE = "js"


class MultiPLE_NUPRL_EN_HumanEval_PHP(_MPLE_HumanEval_Base):
    """MultiPL-E HumanEval in PHP — OLMES variant (nuprl/MultiPL-E, humaneval-php, test split).

    Corresponds to ``multipl_e_humaneval:php::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_HumanEval_PHP"
    MULTIPL_E_LANGUAGE = "php"


class MultiPLE_NUPRL_EN_HumanEval_Rust(_MPLE_HumanEval_Base):
    """MultiPL-E HumanEval in Rust — OLMES variant (nuprl/MultiPL-E, humaneval-rs, test split).

    Corresponds to ``multipl_e_humaneval:rs::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_HumanEval_Rust"
    MULTIPL_E_LANGUAGE = "rs"


class MultiPLE_NUPRL_EN_HumanEval_Bash(_MPLE_HumanEval_Base):
    """MultiPL-E HumanEval in Bash — OLMES variant (nuprl/MultiPL-E, humaneval-sh, test split).

    Corresponds to ``multipl_e_humaneval:sh::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_HumanEval_Bash"
    MULTIPL_E_LANGUAGE = "sh"


class _MPLE_MBPP_Base(_MPLE_Base):
    MULTIPL_E_DATASET_PREFIX = "mbpp"


class MultiPLE_NUPRL_EN_MBPP_Cpp(_MPLE_MBPP_Base):
    """MultiPL-E MBPP in C++ — OLMES variant (nuprl/MultiPL-E, mbpp-cpp, test split).

    Corresponds to ``multipl_e_mbpp:cpp::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_MBPP_Cpp"
    MULTIPL_E_LANGUAGE = "cpp"


class MultiPLE_NUPRL_EN_MBPP_Java(_MPLE_MBPP_Base):
    """MultiPL-E MBPP in Java — OLMES variant (nuprl/MultiPL-E, mbpp-java, test split).

    Corresponds to ``multipl_e_mbpp:java::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_MBPP_Java"
    MULTIPL_E_LANGUAGE = "java"


class MultiPLE_NUPRL_EN_MBPP_JS(_MPLE_MBPP_Base):
    """MultiPL-E MBPP in JavaScript — OLMES variant (nuprl/MultiPL-E, mbpp-js, test split).

    Corresponds to ``multipl_e_mbpp:js::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_MBPP_JS"
    MULTIPL_E_LANGUAGE = "js"


class MultiPLE_NUPRL_EN_MBPP_PHP(_MPLE_MBPP_Base):
    """MultiPL-E MBPP in PHP — OLMES variant (nuprl/MultiPL-E, mbpp-php, test split).

    Corresponds to ``multipl_e_mbpp:php::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_MBPP_PHP"
    MULTIPL_E_LANGUAGE = "php"


class MultiPLE_NUPRL_EN_MBPP_Rust(_MPLE_MBPP_Base):
    """MultiPL-E MBPP in Rust — OLMES variant (nuprl/MultiPL-E, mbpp-rs, test split).

    Corresponds to ``multipl_e_mbpp:rs::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_MBPP_Rust"
    MULTIPL_E_LANGUAGE = "rs"


class MultiPLE_NUPRL_EN_MBPP_Bash(_MPLE_MBPP_Base):
    """MultiPL-E MBPP in Bash — OLMES variant (nuprl/MultiPL-E, mbpp-sh, test split).

    Corresponds to ``multipl_e_mbpp:sh::olmo3:n32:v2`` in oe_eval.
    Recommended: 0-shot, temp=0.6, top_p=0.6, repeats=32.
    """

    NAME = "MultiPLE_NUPRL_EN_MBPP_Bash"
    MULTIPL_E_LANGUAGE = "sh"
