"""Math reasoning benchmarks."""

import re
from collections.abc import Callable
from typing import Any, final, override

from eval_framework.answer import ExtractFromCompletion, Extractor, PickFromCandidates
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice, Generative, ItemText
from eval_framework.fewshot import FewShot, FewshotExample, FunctionRenderer, NoFewShot, Predefined
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.language_checker import LanguageRawConsistencyChecker
from eval_framework.metrics.completion.math_minerva_completion import (
    MathMinervaCompletion,
    MathMinervaCompletionRelaxed,
)
from eval_framework.metrics.completion.math_reasoning_completion import MathReasoningCompletion
from eval_framework.metrics.completion.minerva_math_utils import (
    _fix_a_slash_b,
    _fix_fracs,
    _fix_sqrt,
    _remove_right_units,
    extract_answers,
    normalized_gold_from_solution,
    strip_string_hendrycks,
)
from eval_framework.subjects import ListOfSubjects, NoSubject, SubjectsSelector
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle

MATH500_DATASET_PATH = "HuggingFaceH4/MATH-500"

# From OpenAI's simple-evals math_eval.py (c) 2024 OpenAI, MIT. Continuation lines keep a 4-space indent, as
# in the original; ``{{answer}}`` renders as the literal ``{answer}``.
_QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

    {Question}

    Remember to put your answer in $\\boxed{{answer}}$

    where [answer] is just the final number or expression that solves the problem.
""".strip()  # noqa: E501

_ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"


def _extract_boxed(string: str) -> str | None:
    if "\\boxed" not in string and "\\fbox" not in string:
        return None
    idx = max(string.rfind("\\boxed"), string.rfind("\\fbox"))
    right_brace_idx = None
    num_left_braces_open = 0
    i = idx
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    retval = string[idx : right_brace_idx + 1]
    left = "\\boxed{"
    if retval[: len(left)] == left and retval[-1] == "}":
        return retval[len(left) : -1]
    return None


def _strip_string_with_bug(string: str) -> str:
    """The original MATH answer normalisation, **faithfully preserved with its bug**: the ``\\left`` /
    ``\\right`` / ``^\\circ`` / ``\\!`` entries are raw patterns (``r"\\\\left"`` = *two* backslashes), so
    despite the comments they never match real single-backslash LaTeX and those commands are left in place.
    ``MATH500_V2`` uses the corrected ``strip_string_hendrycks`` instead. Sub-helpers are shared with minerva
    (verified identical)."""
    for pattern, replacement in [
        (r"\n", ""),  # linebreaks
        (r"\\!", ""),  # remove inverse spaces
        (r"\\\\", "\\"),  # replace \\ with \
        (r"tfrac", "frac"),
        (r"dfrac", "frac"),
        (r"\\left", ""),  # remove \left  (bug: matches two backslashes, never a real \left)
        (r"\\right", ""),  # remove \right (same bug)
        (r"^{\\circ}", ""),
        (r"^\\circ", ""),
        (r"\\$", ""),  # remove $
    ]:
        string = string.replace(pattern, replacement)
    string = _remove_right_units(string)
    for pattern, replacement in [(r"\\%", ""), (r"\%", ""), (r" .", " 0."), (r"{.", "{0.")]:  # noqa: W605
        string = string.replace(pattern, replacement)
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    parts = [s.strip() for s in string.split("=")]
    if len(parts) == 2 and len(parts[0]) <= 2:
        string = parts[1]
    elif len(parts) > 2:
        if all(len(part) <= 2 and re.match(r"^[a-zA-Z]\w*$", part) for part in parts[:-1]):  # noqa: W605
            string = parts[-1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    string = re.sub(r"\b0(?=\d)", "", string)  # remove leading zeros
    return string


def _boxed_extractor(strip: Callable[[str], str], *, answer_line_fallback: bool = False) -> Extractor:
    def extract(completion_text: str) -> str:
        answer = _extract_boxed(completion_text)
        if answer is None and answer_line_fallback:
            match = re.search(_ANSWER_PATTERN, completion_text)
            answer = match.group(1) if match else None
        return "[no_answer]" if answer is None else strip(answer)

    return extract


def _math500(id: str, strip: Callable[[str], str], dataset: DatasetPolicy | None) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        kind=Generative(
            build_prompt=lambda item: _QUERY_TEMPLATE.format(Question=item["problem"]),
            cue="",  # no assistant cue — the model produces the full worked solution
            ground_truth=lambda item: item["answer"],
            metrics=[MathReasoningCompletion, LanguageRawConsistencyChecker],
        ),
        answer=ExtractFromCompletion(_boxed_extractor(strip, answer_line_fallback=True)),
        sample_split="test",
        fewshot=NoFewShot(),
        subjects=NoSubject(),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(MATH500_DATASET_PATH),
        language=Language.ENG,
    )


def math500_with_bug(dataset: DatasetPolicy | None = None) -> Benchmark:
    # V1 — faithful to the original MATH500, including the latent normalisation bug in
    # `_strip_string_with_bug`. `math500_v2` is the same task with the bug fixed.
    return _math500("MATH500", _strip_string_with_bug, dataset)


def math500_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    # V2 — identical prompt and extraction, but the corrected shared normalisation, so LaTeX commands like
    # \left / ^\circ are stripped as originally intended.
    return _math500("MATH500_V2", strip_string_hendrycks, dataset)


# From NVIDIA NeMo-Skills (llama3-instruct/math.yaml). Continuation lines keep a 4-space indent, as in the
# original, and there is no trailing strip; ``{{answer}}`` renders as the literal ``{answer}``.
_AIME_QUERY_TEMPLATE = """Solve the following math problem efficiently and clearly:

    - For simple problems (2 steps or fewer):
    Provide a concise solution with minimal explanation.

    - For complex problems (3 steps or more):
    Use this step-by-step format:

    ## Step 1: [Concise description]
    [Brief explanation and calculations]

    ## Step 2: [Concise description]
    [Brief explanation and calculations]

    ...

    Regardless of the approach, always conclude with:

    Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.

    Where [answer] is just the final number or expression that solves the problem.

    Problem: {Question}"""  # noqa: E501


def _aime_default_prompt(item: dict[str, Any]) -> str:
    return _AIME_QUERY_TEMPLATE.format(Question=item["problem"])


def aime(
    id: str,
    *,
    dataset_policy: DatasetPolicy,
    ground_truth: ItemText,
    sample_split: str,
    build_prompt: ItemText = _aime_default_prompt,
    subjects: SubjectsSelector = NoSubject(),
    language: Language = Language.ENG,
) -> Benchmark:
    """Build an AIME-style boxed-answer benchmark: 0-shot generative, boxed extraction with the preserved MATH
    normalisation (its bug is inert on integer answers), scored by ``MathReasoningCompletion``. Callers vary the
    prompt (``build_prompt``, defaulting to the English NeMo-Skills template), dataset, subjects and language —
    e.g. localized AIME variants in the companion package reuse this."""
    return ComposedBenchmark.compose(
        id=id,
        kind=Generative(
            build_prompt=build_prompt,
            cue="",
            ground_truth=ground_truth,
            metrics=[MathReasoningCompletion, LanguageRawConsistencyChecker],
        ),
        answer=ExtractFromCompletion(_boxed_extractor(_strip_string_with_bug)),  # boxed only, no Answer: fallback
        sample_split=sample_split,
        fewshot=NoFewShot(),
        subjects=subjects,
        dataset_policy=dataset_policy,
        language=language,
    )


def aime2024(dataset: DatasetPolicy | None = None) -> Benchmark:
    # AIME 2024 gold answers are zero-padded (range 0-999); strip the leading zeros.
    return aime(
        "AIME2024",
        dataset_policy=dataset if dataset is not None else pinned_by_framework("HuggingFaceH4/aime_2024"),
        ground_truth=lambda item: item["answer"].lstrip("0"),
        sample_split="train",
    )


def aime2025(dataset: DatasetPolicy | None = None) -> Benchmark:
    return aime(
        "AIME2025",
        dataset_policy=dataset if dataset is not None else pinned_by_framework("math-ai/aime25"),
        ground_truth=lambda item: item["answer"],
        sample_split="test",
    )


def aime2026(dataset: DatasetPolicy | None = None) -> Benchmark:
    return aime(
        "AIME2026",
        dataset_policy=dataset if dataset is not None else pinned_by_framework("math-ai/aime26"),
        ground_truth=lambda item: item["answer"],
        sample_split="test",
    )


GSM8K_REASONING_DATASET_PATH = "openai/gsm8k"

# Zero-shot reasoning prompt (boxed final answer). Distinct from the OLMES gsm8k variants in benchmarks/gsm8k.py.
_GSM8K_REASONING_QUERY_TEMPLATE = """\
Solve the following math problem step by step. Think through the problem carefully and show your reasoning.

Please provide your answer in the format: $\\boxed{{answer}}$ where answer is the final numerical result.

Question: {question}

Answer:"""

_HASH_ANSWER_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_hash_answer(text: str) -> str:
    """The GSM8K gold-answer form: the number after ``####`` with commas removed, or ``"[invalid]"``."""
    match = _HASH_ANSWER_PATTERN.search(text)
    return match.group(1).strip().replace(",", "") if match else "[invalid]"


def _gsm8k_reasoning_extractor(completion_text: str) -> str:
    """Boxed answer first (commas stripped); otherwise fall back to the ``####`` line."""
    boxed = _extract_boxed(completion_text)
    if boxed is not None:
        return boxed.replace(",", "").strip()
    return extract_hash_answer(completion_text)


def gsm8k_reasoning(dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id="GSM8KReasoning",
        kind=Generative(
            build_prompt=lambda item: _GSM8K_REASONING_QUERY_TEMPLATE.format(question=item["question"]),
            cue="",  # the prompt already ends on "Answer:"; the model continues from there
            ground_truth=lambda item: extract_hash_answer(item["answer"]),
            metrics=[AccuracyCompletion, LanguageRawConsistencyChecker],
        ),
        answer=ExtractFromCompletion(_gsm8k_reasoning_extractor),  # boxed, then #### fallback; no stop sequences
        sample_split="test",
        fewshot=NoFewShot(),
        subjects=ListOfSubjects(["main"]),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(GSM8K_REASONING_DATASET_PATH),
        language=Language.ENG,
    )


# --- Minerva-style MATH (OLMES): "Problem:/Solution:" prompt, fixed 4-shot block, final-answer matching ---

HENDRYCKS_MATH_DATASET_PATH = "EleutherAI/hendrycks_math"

# Per-subject configs of the Hendrycks MATH dataset.
_MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

_MINERVA_MAX_TOKENS = 1024

# The canonical OLMES 4-shot block (hand-written, not sampled from the dataset).
# https://github.com/huggingface/lm-evaluation-harness/blob/add_leaderboard_tasks/lm_eval/tasks/leaderboard/math/utils.py
_OLMES_FEWSHOTS = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so "
        "$x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$,"
        " which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The "
        "final answer is $[2,5)$. I hope it is correct.",
        "few_shot": "1",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} "
        "\\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})"
        " = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
        "few_shot": "1",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, "
        "how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ "
        "pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of "
        "$2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\"
        "begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer:"
        " The final answer is $16$. I hope it is correct.",
        "few_shot": "1",
    },
    {
        "problem": "If the system of equations\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}\nhas a "
        "solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$"
        "Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac"
        "{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
        "few_shot": "1",
    },
]


def _minerva_prompt(item: dict[str, Any]) -> str:
    return "Problem:\n" + item["problem"] + "\n\nSolution:"


def _minerva_gold(item: dict[str, Any]) -> str:
    # normalized_gold_from_solution returns None on a boxless/malformed solution; "" reads the same as None
    # to the minerva metric (both hit its "no ground truth" branch), so we coalesce to keep the gold a str.
    return normalized_gold_from_solution(item["solution"]) or ""


def _minerva_extractor(completion_text: str) -> str:
    candidates = extract_answers(completion_text, use_cot=True, cot_style="minerva", relaxed=True)
    return candidates[0] if candidates else "[no_answer]"


def _olmes_generative_demo(demo: dict[str, Any]) -> FewshotExample:
    return FewshotExample(prompt=_minerva_prompt(demo), answer=" " + demo["solution"])


def _mathminerva_olmes(id: str, stop_sequences: list[str], dataset: DatasetPolicy | None) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        kind=Generative(
            build_prompt=_minerva_prompt,
            cue="",  # the prompt ends on "Solution:"; the model continues from there
            ground_truth=_minerva_gold,
            metrics=[MathMinervaCompletion, MathMinervaCompletionRelaxed],
        ),
        answer=ExtractFromCompletion(_minerva_extractor, stop_sequences, max_tokens=_MINERVA_MAX_TOKENS),
        sample_split="test",
        fewshot=FewShot(Predefined(_OLMES_FEWSHOTS, count=4, label=id), FunctionRenderer(_olmes_generative_demo)),
        subjects=ListOfSubjects(_MATH_SUBJECTS),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(HENDRYCKS_MATH_DATASET_PATH),
        language=Language.ENG,
    )


def mathminerva_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mathminerva_olmes("MATHMinerva_OLMES", ["Problem:", "\n\n"], dataset)


def mathminerva_olmes_nonl(dataset: DatasetPolicy | None = None) -> Benchmark:
    # Same as MATHMinerva_OLMES but stops only on "Problem:" (drops the blank-line stop).
    return _mathminerva_olmes("MATHMinerva_OLMES_NONL", ["Problem:"], dataset)


_MINERVA_BPB_STYLER = BPBStyle(question_prefix="Problem:\n", cue_text="Solution:")


@final
class _MinervaBpbReader(ChoiceReader):
    """The single scored 'choice' is the gold solution; BPB scores the model's likelihood of it."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question=item["problem"], choices=[item["solution"]], correct_index=0)


_MINERVA_BPB_READER = _MinervaBpbReader()


def _minerva_bpb_demo(demo: dict[str, Any]) -> FewshotExample:
    # Rendered through the same styler as the eval item, so the fixed 4-shot block matches the BPB format.
    fields = _MINERVA_BPB_READER.read(demo)
    return FewshotExample(
        prompt=_MINERVA_BPB_STYLER.get_instruction_text(fields.raw_question, fields.choices),
        answer=_MINERVA_BPB_STYLER.get_fewshot_target_text(fields.choices, fields.correct_index),
    )


def mathminerva_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id="MATHMinervaBPB",
        kind=Choice(_MINERVA_BPB_READER, _MINERVA_BPB_STYLER),
        answer=PickFromCandidates(),
        sample_split="test",
        fewshot=FewShot(
            Predefined(_OLMES_FEWSHOTS, count=4, label="MATHMinervaBPB"), FunctionRenderer(_minerva_bpb_demo)
        ),
        subjects=ListOfSubjects(_MATH_SUBJECTS),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(HENDRYCKS_MATH_DATASET_PATH),
        language=Language.ENG,
    )


MATH_REASONING_BENCHMARKS: list[Benchmark] = [
    math500_with_bug(),
    math500_v2(),
    aime2024(),
    aime2025(),
    aime2026(),
    gsm8k_reasoning(),
    mathminerva_olmes(),
    mathminerva_olmes_nonl(),
    mathminerva_bpb(),
]
