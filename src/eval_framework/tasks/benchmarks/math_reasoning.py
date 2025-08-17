import random
import re
from typing import Any, TypeVar

from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion_metrics.language_checker import LanguageRawConsistencyChecker
from eval_framework.metrics.completion_metrics.math_reasoning_completion import MathReasoningCompletion
from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, BaseTask, Language, ResponseType, Sample

SubjectType = TypeVar("SubjectType")


class MATHReasoning(BaseTask[str]):
    """AIME 2024 dataset: https://huggingface.co/datasets/HuggingFaceH4/aime_2024

    This dataset contains a single train split of 30 questions.
    Data contains
        ID | Problem | Solution | Answer

    pass@1 evaluation
    """

    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [MathReasoningCompletion]
    SUBJECTS = [NO_SUBJECT]
    ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        # Max tokens are going to be determined by the model.
        # however GPT paper and results used 1024 tokens, s1 used 2048

    def _extract_answer(
        self, string: str, extract_from_boxed: bool = True, extract_regex: str = ANSWER_PATTERN
    ) -> str | None:
        """Extract Answer String from \\boxed expression or based on regex"""
        if not extract_from_boxed:
            match = re.search(extract_regex, string)
            if match:
                return match.group(1)
            return None

        if "\\boxed" not in string and "\\fbox" not in string:
            return None

        idx_boxed = string.rfind("\\boxed")
        idx_fbox = string.rfind("\\fbox")
        idx = max(idx_boxed, idx_fbox)

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
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
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        if retval:
            left = "\\boxed{"
            try:
                assert retval[: len(left)] == left
                assert retval[-1] == "}"
                return retval[len(left) : -1]
            except AssertionError:
                return None

        return None

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        assert isinstance(completion_text, str)
        extracted_answer = self._extract_answer(completion_text)
        if extracted_answer is None:
            normalized_answer = "[no_answer]"
        else:
            normalized_answer = self._strip_string(extracted_answer)
        return normalized_answer

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        raise NotImplementedError("This method should be implemented in subclasses")

    # The following code is coming from the Eleuther AI lm-evaluation-harness repository
    # Subject to MIT License

    # This needs a major refactoring but is kept as is for consistency with the original code

    def _find_closing_bracket(self, string: str, start_index: int) -> int:
        """
        Finds the index of the closing '}' for a '{' at the given start index.

        :param string: The input string containing '{' and '}' brackets.
        :param start_index: The index where the opening '{' is located.
        :return: The index of the corresponding closing '}' or -1 if not found.
        """
        if start_index < 0 or start_index >= len(string) or string[start_index] != "{":
            raise ValueError("The start_index must point to a '{' character.")

        depth = 0  # Track the nesting level of brackets
        for i in range(start_index, len(string)):
            if string[i] == "{":
                depth += 1  # Increase depth for each opening bracket
            elif string[i] == "}":
                depth -= 1  # Decrease depth for each closing bracket
                if depth == 0:
                    return i  # Found the matching closing bracket

        return -1  # No matching '}' found

    def _split_text_command(self, string: str, search: str = r"\text{") -> tuple[str, str, str]:
        """
        Extracts the content inside a LaTeX \text{...} command and returns three parts:

        1. Everything before `\text{`
        2. The content inside `\text{...}`
        3. Everything after the closing `}`

        :param string: The input LaTeX string.
        :param search: The command to search for (default: `\text{`).
        :return: Tuple (before_text, inside_text, after_text).
                If no `\text{}` is found, returns (string, "", "").
                If no closing bracket `}` is found, returns (before_text, remaining_string, "").
        """
        search_len = len(search)
        search_start = string.find(search)

        # If \text{ is not found, return the entire string in `before_text`
        if search_start == -1:
            return string, "", ""

        # Ensure `{` follows the search term
        content_start = search_start + search_len - 1
        if content_start >= len(string) or string[content_start] != "{":
            return string, "", ""

        # Find the corresponding closing bracket
        closing_index = self._find_closing_bracket(string, start_index=content_start)

        # If no closing bracket is found, return remaining string as "inside_text"
        if closing_index == -1:
            return string[:search_start], string[content_start + 1 :], ""

        before_text = string[:search_start]  # Everything before `\text{`
        inside_text = string[content_start + 1 : closing_index]  # Content inside `\text{...}`
        after_text = string[closing_index + 1 :]  # Everything after the closing `}`

        return before_text, inside_text, after_text

    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L144
    def _remove_right_units(self, string: str) -> str:
        # "\text{ " only ever occurs (at least in the val set) when describing units
        count = string.count(r"\text{")
        if count == 0:
            return string
        elif count > 1:
            content, *_ = string.split(r"\text{", maxsplit=1)
            return content
        elif count == 1:
            before, inside, after = self._split_text_command(string)
            if before.strip():
                return before.strip()
            elif after.strip():
                return after.strip()
            else:
                return inside.strip()
        else:
            raise ValueError("Unexpected count of units in string")

    # Based on https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L154
    def _fix_sqrt(self, string: str) -> str:
        if "\\sqrt" not in string:
            return string
        parts = string.split("\\sqrt")
        new_string = parts[0]
        for part in parts[1:]:
            new_string += "\\sqrt{"
            if part[0] != "{":
                new_string += part[0] + "}"
            new_string += part[1:]
        return new_string

    # Based on https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L97
    def _fix_fracs(self, string: str) -> str:
        parts = string.split("\\frac")
        if len(parts) <= 1:
            return string
        new_str = parts[0]
        for part in parts[1:]:
            new_str += "\\frac"
            if not part:
                continue
            if part[0] == "{":
                new_str += part
            else:
                try:
                    assert len(part) >= 2
                except AssertionError:
                    return string
                a = part[0]
                b = part[1]
                new_str += "{" + a + "}{"
                if b != "{":
                    new_str += b + "}"

                if len(part) > 2:
                    new_str += part[2:]
        return new_str

    # Based on https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L129
    def _fix_a_slash_b(self, string: str) -> str:
        if len(string.split("/")) != 2:
            return string
        a, b = string.split("/")
        try:
            a_int = int(a)
            b_int = int(b)
            assert string == f"{a_int}/{b_int}"
            new_string = "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
            return new_string
        except AssertionError:
            return string
        except ValueError:
            return string

    # Based on https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L169
    def _strip_string(self, string: str) -> str:
        replacements = [
            (r"\n", ""),  # linebreaks
            (r"\\!", ""),  # remove inverse spaces
            (r"\\\\", "\\"),  # replace \\ with \
            (r"tfrac", "frac"),  # replace tfrac with frac
            (r"dfrac", "frac"),  # replace dfrac with frac
            (r"\\left", ""),  # remove \left
            (r"\\right", ""),  # remove \right
            (r"^{\\circ}", ""),  # remove circ
            (r"^\\circ", ""),  # remove circ
            (r"\\$", ""),  # remove $
        ]
        for pattern, replacement in replacements:
            string = string.replace(pattern, replacement)

        # remove units (on the right)
        string = self._remove_right_units(string)

        replacements = [
            (r"\\%", ""),  # remove percentage
            (r"\%", ""),  # noqa: W605 # remove percentage
            (r" .", " 0."),  # " 0." equivalent to " ."
            (r"{.", "{0."),  # "{0." equivalent to "{."
        ]
        for pattern, replacement in replacements:
            string = string.replace(pattern, replacement)

        # if empty, return empty string
        if len(string) == 0:
            return string
        # Add "0" if "." is the start of the string
        if string[0] == ".":
            string = "0" + string
        # Get rid of e.g. "k = " or "x = y = " at beginning
        parts = [s.strip() for s in string.split("=")]
        if len(parts) == 2 and len(parts[0]) <= 2:
            string = parts[1]
        elif len(parts) > 2:
            if all(len(part) <= 2 and re.match(r"^[a-zA-Z]\w*$", part) for part in parts[:-1]):  # noqa: W605
                string = parts[-1]

        # fix sqrt3 --> sqrt{3}
        string = self._fix_sqrt(string)

        # remove spaces
        string = string.replace(r" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2},
        # etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        def strip_leading_zero(s: str) -> str:
            """strip leading zeros, but keep the first zero if it is a decimal"""
            return re.sub(r"\b0(?=\d)", "", s)

        # remove leading zeros
        string = strip_leading_zero(string)

        return string


class AIME2024(MATHReasoning):
    """AIME 2024 dataset: https://huggingface.co/datasets/HuggingFaceH4/aime_2024

    This dataset contains a single train split of 30 questions.
    Data contains
        ID | Problem | Solution | Answer

    pass@1 evaluation
    """

    NAME = "AIME2024"
    DATASET_PATH = "HuggingFaceH4/aime_2024"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [MathReasoningCompletion, LanguageRawConsistencyChecker]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    # https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/prompt/config/llama3-instruct/math.yaml
    QUERY_TEMPLATE = """Solve the following math problem efficiently and clearly:

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
    ANSWER_PATTERN = r"Therefore, the final answer is:(.*?). I hope it is correct."

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "AIME evaluation does not include few shot"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return self.QUERY_TEMPLATE.format(Question=item["problem"])

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        return item["answer"].lstrip("0")  # valid answers in this dataset range from 0-999 and have leading zeros


class MATH500(MATHReasoning):
    """MATH500 dataset: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

    This dataset contains a single test split of 500 questions.
    Data contains

        ID | Problem | Solution | Answer

    pass@1 evaluation
    """

    NAME = "MATH500"
    DATASET_PATH = "HuggingFaceH4/MATH-500"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [MathReasoningCompletion, LanguageRawConsistencyChecker]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    # Adapted from https://github.com/openai/simple-evals/blob/main/math_eval.py
    QUERY_TEMPLATE = """
    Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

    {Question}

    Remember to put your answer in $\\boxed{{answer}}$

    where [answer] is just the final number or expression that solves the problem.
    """.strip()  # noqa: E501

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "MATH-500 evaluation does not include few shot"
        super().__init__(num_fewshot)

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        extracted_answer_boxed = self._extract_answer(completion_text)
        extracted_answer_unboxed = self._extract_answer(
            completion_text, extract_from_boxed=False, extract_regex=self.ANSWER_PATTERN
        )
        # if there is no "boxed" answer but there is an "Answer: " answer, use the latter
        extracted_answer = extracted_answer_boxed if extracted_answer_boxed is not None else extracted_answer_unboxed
        if extracted_answer is None:
            normalized_answer = "[no_answer]"
        else:
            normalized_answer = self._strip_string(extracted_answer)
        return normalized_answer

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return self.QUERY_TEMPLATE.format(Question=item["problem"])

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        return item["answer"]


class MATH(MATHReasoning):
    """MATH dataset: https://huggingface.co/datasets/EleutherAI/hendrycks_math"""

    NAME = "Math"
    DATASET_PATH = "EleutherAI/hendrycks_math"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [MathReasoningCompletion, LanguageRawConsistencyChecker]
    SUBJECTS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["\nProblem:", "\nProblem", "\n\nProblem:", "\n\nProblem"]

    def extract_last_two_dollar_text(self, s: str) -> str:
        """
        extract_last_two_dollar_text finds text between the last two dollar signs in a string
        :param s: the string to extract text from
        :returns: the extracted text
        """
        finds = re.findall(r"\$(.*?)\$", s)
        match = "" if len(finds) == 0 else finds[-1]
        return match

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """
        post_process_generated_completion extracts via flex extraction/matching.
        if there is a boxed answer, then this gets used first
        if there is no boxed answer, and latex math symbols ("$") then this will be extracted and used
        if there is an answer text ("Answer:") then this will be used last

        """
        extracted_answer_boxed = self._extract_answer(completion_text)
        extracted_answer_latex_math_symb = self._extract_answer(self.extract_last_two_dollar_text(completion_text))
        extracted_answer_unboxed = self._extract_answer(
            completion_text, extract_from_boxed=False, extract_regex=self.ANSWER_PATTERN
        )
        # if there is no "boxed" answer but there is an "Answer: " answer, use the latter
        if extracted_answer_boxed:
            normalized_answer = self._strip_string(extracted_answer_boxed)
        elif extracted_answer_latex_math_symb:
            normalized_answer = self._strip_string(extracted_answer_latex_math_symb)
        elif extracted_answer_unboxed:
            normalized_answer = self._strip_string(extracted_answer_unboxed)
        else:
            normalized_answer = "[no_answer]"
        return normalized_answer

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Problem: {item['problem']}\nAnswer: "

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return item["solution"]

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        return self._extract_answer(item["solution"])


class MATHLvl5(MATH):
    NAME = "Math Lvl 5"

    def _load_dataset(self, subject: SubjectType) -> None:
        name = subject if subject != NO_SUBJECT else None

        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=name)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = [item for item in data_list if item["level"] == "Level 5"]

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        return self._extract_answer(item["solution"])


class GSM8KReasoning(MATHReasoning):
    """GSM8K dataset with reasoning prompt: https://huggingface.co/datasets/openai/gsm8k

    Zero-shot reasoning version that expects answers in boxed format.
    """

    NAME = "GSM8KReasoning"
    DATASET_PATH = "gsm8k"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, LanguageRawConsistencyChecker]
    SUBJECTS = ["main"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    # Reasoning prompt template that encourages step-by-step thinking with boxed answers
    QUERY_TEMPLATE = """\
Solve the following math problem step by step. Think through the problem carefully and show your reasoning.

Please provide your answer in the format: $\\boxed{{answer}}$ where answer is the final numerical result.

Question: {question}

Answer:"""

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "GSM8K Reasoning is designed for zero-shot evaluation only"
        super().__init__(num_fewshot)
        self.stop_sequences: list[str] = []

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return self._extract_answer_with_fallback(completion_text)

    def _extract_answer_fallback(self, completion: str) -> str:
        """Fallback answer extraction using #### pattern for compatibility"""
        ans_re = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ans_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return "[invalid]"

    def _extract_answer_with_fallback(self, completion: str) -> str:
        """Extract answer from completion, trying boxed format first, then fallback"""
        # Try boxed format first
        boxed_answer = self._extract_answer(completion)
        if boxed_answer is not None:
            # Clean the answer by removing commas and whitespace
            cleaned_answer = boxed_answer.replace(",", "").strip()
            return cleaned_answer

        # Fallback to #### pattern
        return self._extract_answer_fallback(completion)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return self.QUERY_TEMPLATE.format(question=item["question"])

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        # This should not be called since we're zero-shot only
        raise NotImplementedError("GSM8K Reasoning is zero-shot only")

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return self._extract_answer_fallback(item["answer"])
