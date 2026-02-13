"""
Minerva-style MATH answer extraction and equivalence (Lewkowycz et al. 2022).
"""

import re
import signal
from typing import Any

from sympy import SympifyError, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

INVALID_ANSWER = "[invalidanswer]"
END_SEQ = "I hope it is correct."

# Minerva normalize_final_answer: appendix D of Lewkowycz et al. (2022)
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} from string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove \\boxed{ or \\boxed from content."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def get_unnormalized_answer(text: str) -> str:
    """Extract answer from Minerva 'Final Answer: The final answer is ... I hope it is correct.'"""
    text = text + END_SEQ
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer (appendix D of Lewkowycz et al. 2022).
    """
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def _find_closing_bracket(string: str, start_index: int) -> int:
    depth = 0
    for i in range(start_index, len(string)):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_text_command(string: str, search: str = r"\text{") -> tuple[str, str, str]:
    search_len = len(search)
    search_start = string.find(search)
    if search_start == -1:
        return string, "", ""
    content_start = search_start + search_len - 1
    if content_start >= len(string) or string[content_start] != "{":
        return string, "", ""
    closing_index = _find_closing_bracket(string, content_start)
    if closing_index == -1:
        return string[:search_start], string[content_start + 1 :], ""
    before_text = string[:search_start]
    inside_text = string[content_start + 1 : closing_index]
    after_text = string[closing_index + 1 :]
    return before_text, inside_text, after_text


def _remove_right_units(string: str) -> str:
    if r"\text{" not in string:
        return string
    if string.count(r"\text{") > 1:
        return string.split(r"\text{", maxsplit=1)[0]
    before, inside, after = _split_text_command(string)
    if before.strip():
        return before.strip()
    if after.strip():
        return after.strip()
    return inside.strip()


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    parts = string.split("\\sqrt")
    new_string = parts[0]
    for part in parts[1:]:
        new_string += "\\sqrt{"
        if part and part[0] != "{":
            new_string += part[0] + "}"
        new_string += part[1:] if len(part) > 1 else ""
    return new_string


def _fix_fracs(string: str) -> str:
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
            if len(part) < 2:
                return string
            a, b = part[0], part[1]
            new_str += "{" + a + "}{"
            if b != "{":
                new_str += b + "}"
            if len(part) > 2:
                new_str += part[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int, b_int = int(a), int(b)
        if string == f"{a_int}/{b_int}":
            return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
    except (AssertionError, ValueError):
        pass
    return string


def strip_string_hendrycks(string: str) -> str:
    """Hendrycks-style string normalization for string equivalence."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    parts = [s.strip() for s in string.split("=")]
    if len(parts) == 2 and len(parts[0]) <= 2:
        string = parts[1]
    elif len(parts) > 2:
        if all(len(part) <= 2 and re.match(r"^[a-zA-Z]\w*$", part) for part in parts[:-1]):
            string = parts[-1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    string = re.sub(r"\b0(?=\d)", "", string)
    return string


def is_equiv_minerva(x1: str, x2: str, timeout_seconds: int = 5) -> bool:
    """Sympy-based equivalence (Minerva)."""

    def _timeout_handler(signum: Any, frame: Any) -> None:
        raise TimeoutError()

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            parsed_x1 = parse_latex(x1)
            parsed_x2 = parse_latex(x2)
        except (LaTeXParsingError, SympifyError, TypeError):
            return False
        try:
            diff = parsed_x1 - parsed_x2
        except TypeError:
            return False
        try:
            return simplify(diff) == 0
        except (ValueError, TimeoutError):
            return False
        finally:
            signal.alarm(0)
    except Exception:
        return False
    finally:
        signal.alarm(0)


def is_equiv_hendrycks(str1: str | None, str2: str | None) -> bool:
    """String equality after Hendrycks strip_string."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return strip_string_hendrycks(str1) == strip_string_hendrycks(str2)
    except Exception:
        return str1 == str2


def extract_answers(
    raw_answer: str,
    use_cot: bool = True,
    cot_style: str = "minerva",
) -> list[str]:
    """
    Extract multiple candidate answers from model output (for exact_match and exact_match_flex).
    Returns list of normalized strings; first is primary for exact_match.
    """
    raw = raw_answer if isinstance(raw_answer, str) else str(raw_answer)
    all_answers: list[str] = []

    if use_cot:
        if cot_style == "minerva":
            minerva_answer = normalize_final_answer(get_unnormalized_answer(raw))
            if minerva_answer and minerva_answer != INVALID_ANSWER:
                all_answers.append(minerva_answer)
        boxed = last_boxed_only_string(raw)
        if boxed is not None:
            try:
                unboxed = remove_boxed(boxed)
                all_answers.append(normalize_final_answer(unboxed))
            except AssertionError:
                pass
        if len(all_answers) == 0:
            dollars = [m.start() for m in re.finditer(r"\$", raw)]
            if len(dollars) > 1:
                all_answers.append(normalize_final_answer(raw[dollars[-2] + 1 : dollars[-1]]))
        if len(all_answers) == 0:
            all_answers.append(normalize_final_answer(raw))
    else:
        all_answers.append(normalize_final_answer(raw))
        dollars = [m.start() for m in re.finditer(r"\$", raw)]
        if len(dollars) > 1:
            all_answers.append(normalize_final_answer(raw[dollars[0] + 1 : dollars[1]]))

    return all_answers
