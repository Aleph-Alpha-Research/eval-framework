"""
Minerva-style MATH answer extraction and equivalence (Lewkowycz et al. 2022).
"""

import atexit
import multiprocessing
import os
import re
import threading
from multiprocessing.context import DefaultContext
from typing import cast

from sympy import SympifyError, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

# Fork inherits an already-imported SymPy in the child (fast). Spawn re-imports the
# whole stack per worker and is much slower; we only use it on Windows where fork
# is unavailable. See https://docs.python.org/3/library/multiprocessing.html#contexts
# Stubs type get_context() as BaseContext, which omits Process; runtime is DefaultContext.
_worker_ctx = cast(
    DefaultContext,
    multiprocessing.get_context("spawn" if os.name == "nt" else "fork"),
)

# Virtual-address-space budget for a single sympy simplify() call.
# Pathological expressions can cause sympy to allocate tens of GiBs;
# this cap turns that into a caught MemoryError instead of an OOM kill.
_SYMPY_MEMORY_BUDGET_BYTES = 2 * 1024**3  # 2 GiB

INVALID_ANSWER = "[invalidanswer]"
END_SEQ = "I hope it is correct."
END_SEQ_DE = "Ich hoffe, die Antwort ist korrekt."  # German pendant to END_SEQ

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


# Registry of supported `cot_style` values
# Keys are the strings passed by metric configurations; values are language-specific final-answer extractors
# `(text: str, relaxed: bool) -> str`. Extend this dict to add a new language.
COT_EXTRACTORS = {
    "minerva": get_unnormalized_answer,
    "minerva_de": get_unnormalized_answer_de,
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


def _set_memory_budget(budget_bytes: int) -> None:
    """Limit virtual address space growth to *budget_bytes* above current usage.

    Uses RLIMIT_AS on Linux; silently skipped on unsupported platforms.
    """
    try:
        import resource

        with open(f"/proc/{os.getpid()}/statm") as f:
            current_pages = int(f.read().split()[0])
        page_size = os.sysconf("SC_PAGE_SIZE")
        limit = current_pages * page_size + budget_bytes
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def _sympy_equiv_once(x1: str, x2: str) -> bool:
    """Run Minerva sympy equivalence in the current process (subprocess worker only)."""
    _set_memory_budget(_SYMPY_MEMORY_BUDGET_BYTES)
    try:
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
    except Exception:
        # Catches MemoryError (from RLIMIT_AS) and anything else.
        return False


def _persistent_worker_main(conn: "multiprocessing.connection.Connection") -> None:
    """Long-lived worker: recv (x1, x2), send bool. Exit on None or broken pipe."""
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        x1, x2 = msg
        result = _sympy_equiv_once(x1, x2)
        try:
            conn.send(result)
        except (BrokenPipeError, OSError):
            break
    try:
        conn.close()
    except Exception:
        pass


# One subprocess serves many equivalence checks; restart on timeout, crash, or job limit.
_worker_lock = threading.Lock()
_worker_proc: "multiprocessing.Process | None" = None
_worker_conn: "multiprocessing.connection.Connection | None" = None
_worker_jobs: int = 0
# Periodic worker restart limits SymPy heap growth; keep high to avoid spawn cold-start.
_WORKER_MAX_JOBS_BEFORE_ROTATE = 50_000


def _stop_worker_locked() -> None:
    """Tear down the persistent worker; caller must hold _worker_lock."""
    global _worker_proc, _worker_conn, _worker_jobs
    if _worker_conn is not None:
        # Do not send a shutdown message: the worker may be stuck in simplify()
        # and would not read the pipe until it finishes.
        try:
            _worker_conn.close()
        except Exception:
            pass
        _worker_conn = None
    if _worker_proc is not None:
        if _worker_proc.is_alive():
            _worker_proc.kill()
            _worker_proc.join(timeout=2)
        _worker_proc = None
    _worker_jobs = 0


def _spawn_worker_locked() -> None:
    """Start a fresh worker; caller must hold _worker_lock and no worker alive."""
    global _worker_proc, _worker_conn
    parent_conn, child_conn = _worker_ctx.Pipe(duplex=True)
    p = _worker_ctx.Process(target=_persistent_worker_main, args=(child_conn,))
    p.start()
    child_conn.close()
    _worker_proc = p
    _worker_conn = parent_conn


def _ensure_worker_locked() -> None:
    """Ensure a live worker process; caller must hold _worker_lock."""
    global _worker_jobs
    if _worker_proc is not None and _worker_proc.is_alive() and _worker_conn is not None:
        return
    _stop_worker_locked()
    _spawn_worker_locked()


def _ipc_sympy_equiv(x1: str, x2: str, timeout_seconds: int) -> bool:
    """Send one job to the persistent worker; restart worker on timeout or I/O failure."""
    global _worker_jobs
    _ensure_worker_locked()
    assert _worker_conn is not None and _worker_proc is not None
    try:
        _worker_conn.send((x1, x2))
        poll_timeout = timeout_seconds if timeout_seconds > 0 else 0.0
        if _worker_conn.poll(poll_timeout):
            result = _worker_conn.recv()
            _worker_jobs += 1
            if _worker_jobs >= _WORKER_MAX_JOBS_BEFORE_ROTATE:
                _stop_worker_locked()
            return bool(result)
        # Worker stuck in simplify — discard process.
        _stop_worker_locked()
        return False
    except (EOFError, OSError, BrokenPipeError):
        _stop_worker_locked()
        return False


def _shutdown_persistent_worker() -> None:
    with _worker_lock:
        _stop_worker_locked()


atexit.register(_shutdown_persistent_worker)


# @lru_cache(maxsize=8192)
def _equiv_minerva_cached(x1: str, x2: str, timeout_seconds: int) -> bool:
    """Memoized sympy path (invoked only on LRU miss)."""
    with _worker_lock:
        return _ipc_sympy_equiv(x1, x2, timeout_seconds)


def is_equiv_minerva(x1: str, x2: str, timeout_seconds: int = 5) -> bool:
    """Sympy-based equivalence (Minerva).

    Uses a long-lived subprocess with a capped virtual-address-space budget so
    pathological expressions cannot OOM-kill the parent. Results are memoized
    to avoid repeated IPC for identical (x1, x2, timeout) pairs.
    """
    if x1 == x2:
        return True
    if x1.strip() == x2.strip():
        return True
    return _equiv_minerva_cached(x1, x2, timeout_seconds)


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
