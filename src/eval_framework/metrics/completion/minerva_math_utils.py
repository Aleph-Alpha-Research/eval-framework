"""
Minerva-style MATH answer extraction and equivalence (Lewkowycz et al. 2022).
"""

import multiprocessing
import re
import resource
import threading
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from sympy import simplify
from sympy.parsing.latex import parse_latex

INVALID_ANSWER = "[invalidanswer]"
END_SEQ = "I hope it is correct."
END_SEQ_DE = "Ich hoffe, die Antwort ist korrekt."  # German pendant to END_SEQ
END_SEQ_FR = "J'espère que la réponse est correcte."  # French pendant to END_SEQ
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


def get_unnormalized_answer(text: str, relaxed: bool = False) -> str:
    """Extract answer from Minerva 'Final Answer: The final answer is ... I hope it is correct.'

    When relaxed=False, pattern matches lm-evaluation-harness (lm_eval.tasks.minerva_math.utils)
    for parity: exact capitalization, no flexible whitespace.
    When relaxed=True, accepts any capitalisation of:
      "Final Answer: The answer is " / "Final Answer: The final answer is "
      "The Final Answer: The answer is " / "The Final Answer: The final answer is "
    with flexible whitespace; no suffix required but "I hope it is correct." is stripped when present).
    """
    if relaxed:
        # Case-insensitive; optional "The " prefix; "answer" or "final answer" before "is"
        match = re.search(
            r"(?i)(?:the\s+)?final\s+answer\s*:\s*the\s+(?:final\s+)?answer\s+is\s*(.*)",
            text,
            re.DOTALL,
        )
        if match:
            raw = match.group(1).strip()
            # Strip the optional "I hope it is correct." phrase when present
            raw = re.sub(r"\.?\s*i\s+hope\s+it\s+is\s+correct\.?\s*$", "", raw, flags=re.IGNORECASE).strip()
            return raw
        return INVALID_ANSWER
    text = text + END_SEQ
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


def get_unnormalized_answer_de(text: str, relaxed: bool = False) -> str:
    """German analogue of ``get_unnormalized_answer``."""
    if relaxed:
        match = re.search(
            r"(?i)(?:finale|endgültige)\s+antwort\s*:\s*"
            r"(?:die\s+(?:finale\s+|endgültige\s+)?antwort\s+(?:ist|lautet)\s*)?(.*)",
            text,
            re.DOTALL,
        )
        if match:
            raw = match.group(1).strip()
            raw = re.sub(
                r"\.?\s*ich\s+hoffe,?\s+(?:die\s+antwort|sie|es)\s+(?:ist|sei)\s+korrekt\.?\s*$",
                "",
                raw,
                flags=re.IGNORECASE,
            ).strip()
            return raw
        return INVALID_ANSWER
    text = text + END_SEQ_DE
    match = re.search(
        r"Finale Antwort: Die finale Antwort lautet(.*?)\. Ich hoffe, die Antwort ist korrekt\.",
        text,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


def get_unnormalized_answer_fr(text: str, relaxed: bool = False) -> str:
    """French analogue of ``get_unnormalized_answer``.

    Strict form: ``Réponse finale : La réponse finale est $…$. J'espère que la réponse est correcte.`` (French
    typography puts a space before the colon, as in the other EllaMind French prompts). Relaxed form accepts any
    capitalisation, missing accents, the colon with or without the space, an optional ``La`` before ``réponse
    finale``, an optional ``La réponse (finale) est`` lead-in, and strips a trailing hope phrase such as
    ``J'espère que la réponse est correcte`` / ``J'espère que c'est correct``.
    """
    if relaxed:
        match = re.search(
            r"(?i)(?:la\s+)?r[ée]ponse\s+finale\s*:\s*(?:la\s+r[ée]ponse\s+(?:finale\s+)?est\s*)?(.*)",
            text,
            re.DOTALL,
        )
        if match:
            raw = match.group(1).strip()
            raw = re.sub(
                r"\.?\s*j['’]\s*esp[èe]re\s+qu(?:e|['’])\s*(?:la\s+r[ée]ponse|elle|ce|c['’])\s*(?:est|soit)\s+correcte?\.?\s*$",
                "",
                raw,
                flags=re.IGNORECASE,
            ).strip()
            return raw
        return INVALID_ANSWER
    text = text + END_SEQ_FR
    match = re.search(
        r"Réponse finale : La réponse finale est(.*?)\. J'espère que la réponse est correcte\.",
        text,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


# Registry of supported `cot_style` values
# Keys are the strings passed by metric configurations; values are language-specific final-answer extractors
# `(text: str, relaxed: bool) -> str`. Extend this dict to add a new language.
COT_EXTRACTORS = {
    "minerva": get_unnormalized_answer,
    "minerva_de": get_unnormalized_answer_de,
    "minerva_fr": get_unnormalized_answer_fr,
}


def normalized_gold_from_solution(solution: str) -> str | None:
    """Extract and normalize the gold answer from a solution string (last \\boxed{...})."""
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    try:
        unboxed = remove_boxed(boxed)
    except AssertionError:
        return None
    return normalize_final_answer(unboxed)


def _normalize_latex_core(s: str) -> str:
    """Shared LaTeX normalization (equation extraction, \\text/\\boxed unwrap, frac/sqrt fix, $ strip)."""
    s = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", s)
    s = re.sub(r"(\\text\{)(.*?)(\})", r"\2", s)
    s = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", s)
    s = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", s)
    s = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", s)
    s = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", s)
    s = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", s)
    s = s.replace("$", "")
    return s


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer (appendix D of Lewkowycz et al. 2022).
    """
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = _normalize_latex_core(final_answer)
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


# MATH gold answers stay under ~80 chars.
_PLAUSIBLE_ANSWER_LEN = 128


def _sympy_equiv(x1: str, x2: str) -> bool:
    try:
        diff = parse_latex(x1) - parse_latex(x2)
        return bool(simplify(diff) == 0)
    except Exception:
        return False


_SYMPY_MEMORY_LIMIT_BYTES = 4 * 1024**3


def _serve_sympy_equiv(connection: Connection, memory_limit: int) -> None:
    # RLIMIT_DATA covers the heap and anonymous mappings, where a runaway
    # integer grows, but not the libraries mapped at startup.
    _, hard_limit = resource.getrlimit(resource.RLIMIT_DATA)
    if hard_limit != resource.RLIM_INFINITY:
        memory_limit = min(memory_limit, hard_limit)
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, memory_limit))
    except ValueError:
        # macOS rejects the call and wouldn't enforce the limit; the timeout still bounds the damage.
        pass
    while True:
        try:
            x1, x2 = connection.recv()
        except EOFError:
            return
        connection.send(_sympy_equiv(x1, x2))


class _SympyWorker:
    """Runs sympy equivalence checks in a child process capped in memory and wall time.

    Sympy evaluates eagerly inside C code, e.g. `(23!)!` from a `23!!` candidate, where
    no signal or timeout can interrupt it and memory grows by tens of GiB per second.
    The child takes the hit: the parent kills it and starts a fresh one on the next call.
    """

    def __init__(self, memory_limit: int = _SYMPY_MEMORY_LIMIT_BYTES) -> None:
        self._memory_limit = memory_limit
        self._lock = threading.Lock()
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None

    def is_equiv(self, x1: str, x2: str, timeout_seconds: float) -> bool:
        with self._lock:
            connection = self._connect()
            try:
                connection.send((x1, x2))
                if connection.poll(timeout_seconds):
                    return bool(connection.recv())
            except (EOFError, OSError):
                pass
            self._stop()
            return False

    def _connect(self) -> Connection:
        if self._connection is None or self._process is None or not self._process.is_alive():
            self._stop()
            # spawn, not fork: the parent may be running threads, and the child only needs sympy.
            context = multiprocessing.get_context("spawn")
            self._connection, child_connection = context.Pipe()
            self._process = context.Process(
                target=_serve_sympy_equiv, args=(child_connection, self._memory_limit), daemon=True
            )
            self._process.start()
            child_connection.close()
        return self._connection

    def _stop(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.kill()
            self._process.join()
        if self._connection is not None:
            self._connection.close()
        self._process = None
        self._connection = None


_sympy_worker = _SympyWorker()


def is_equiv_minerva(x1: str, x2: str, timeout_seconds: float = 5) -> bool:
    """Sympy-based equivalence (Minerva)."""
    # extract_answers can fall back to the whole completion; sympy grinds on such
    # 1000+ char strings until the timeout. A side far longer than the other can't
    # be an equivalent final answer, so refuse it before parsing.
    if max(len(x1), len(x2)) > max(_PLAUSIBLE_ANSWER_LEN, 3 * min(len(x1), len(x2))):
        return False
    return _sympy_worker.is_equiv(x1, x2, timeout_seconds)


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
    relaxed: bool = False,
) -> list[str]:
    """
    Extract multiple candidate answers from model output (for exact_match and exact_match_flex).
    Returns list of normalized strings; first is primary for exact_match.
    When relaxed=True, final-answer string matching is more lenient (whitespace/case).
    """
    raw = raw_answer if isinstance(raw_answer, str) else str(raw_answer)
    all_answers: list[str] = []

    if use_cot:
        if cot_style not in COT_EXTRACTORS:
            raise ValueError(f"Unknown cot_style {cot_style!r}; valid: {sorted(COT_EXTRACTORS)}")
        extractor = COT_EXTRACTORS[cot_style]
        minerva_answer = normalize_final_answer(extractor(raw, relaxed=relaxed))
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
