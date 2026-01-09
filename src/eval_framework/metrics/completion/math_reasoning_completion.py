import re
import signal
from collections.abc import Callable, Iterable
from typing import Any

from sympy import Basic, S, SympifyError, factor, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


def timeout_handler(signum: Any, frame: Any) -> None:
    raise TimeoutError()


class MathReasoningCompletion(BaseMetric[Completion]):
    #
    # Math Reasoning Completion (symbolic)
    #
    # This metric evaluates the correctness of the completion of a math reasoning task without
    # correcting LaTeX expressions. Normalization occurs on the strings, only to remove formatting
    # and units.
    #
    # The metric is designed to evaluate the correctness of the completion of a math reasoning task
    # without correcting LaTeX expressions.
    #

    NAME = "Math Reasoning Completion (symbolic)"

    # Substitutions to apply to the final answer
    SUBSTITUTIONS = [
        (r"\ban\b(?!\w)", ""),  # Remove "an" if not part of a word
        (r"\ba\b(?!\w)", ""),  # Remove "a" if not part of a word
        (r"\.\$", "$"),  # Replace ".$" with "$"
        (r"\\\$", ""),  # Remove "\$"
        (r"\\ ", ""),  # Remove "\ " (escaped space)
        (r"\s+", ""),  # Remove all spaces
        (r"\\mbox", "text"),  # Replace "\mbox" with "text"
        (r",\\text\{and\}", ","),  # Replace ",\text{and}" with ","
        (r"\\text\{and\}", ","),  # Replace "\text{and}" with ","
        (r"\\text\{m\}", "\\text{}"),  # Replace "\text{m}" with "\text{}"
    ]

    # Expressions to remove from the final answer
    # Most of these expressions omit units and formatting
    # which the ground truth does not have
    REMOVED_EXPRESSIONS_UNITS = [
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
    ]

    REMOVED_EXPRESSIONS_FORMAT = [
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

    def normalize_expression(self, final_answer: str) -> str:
        """
        Function to normalize LaTeX expressions
        :param final_answer: raw LaTeX expression
        :return: normalized LaTeX expression
        NOTE: Changed logic, because before the substitution randomly replaced characters in the string,
        i.e., turned "infty" into "iny" by removing "ft"
        """
        for before, after in self.SUBSTITUTIONS:
            final_answer = re.sub(before, after, final_answer)
        for expr in self.REMOVED_EXPRESSIONS_UNITS:
            # Safely remove units at the end, allowing optional space before the unit
            final_answer = re.sub(rf"(.*?)\s*({re.escape(expr)})$", r"\1", final_answer)
        for expr in self.REMOVED_EXPRESSIONS_FORMAT:
            # Safely remove formatting expressions
            final_answer = final_answer.replace(expr, "")
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
        final_answer = final_answer.replace("$", "")
        # Only strip commas if it's a single numeric value with optional commas (like "1,000")
        if re.fullmatch(r"\d{1,3}(,\d{3})*", final_answer):
            final_answer = final_answer.replace(",", "")
        return final_answer

    def check_for_equation(self, final_answer: str) -> list:
        """
        Check if the final answer is an equation and split it into left hand side and right hand side
        :param final_answer: the expression to evaluate
        :return: list of left hand side and right hand side of the equation
        """
        if isinstance(final_answer, str) and "=" in final_answer:
            return final_answer.split("=")
        else:
            return [final_answer]

    def _safe_simplify_expression(self, expression: Basic, timeout: int = 10) -> Basic:
        """
        Simplify an expression with a timeout and catch recursion depth exception
        :param expression: SymPy expression
        :param timeout: Time limit in seconds (default: 10 seconds).
        :return: simplified expressions
        """
        signal.signal(signal.SIGALRM, timeout_handler)  # Set timeout signal
        signal.alarm(timeout)  # Set timeout duration

        try:
            factored = factor(expression)
            simplified = simplify(factored)
            return simplified
        except (SympifyError, TimeoutError):
            return S.NaN
        finally:
            # Ensure we never leak a pending alarm into later code paths.
            signal.alarm(0)

    def _any_symb_correct(self, response_list: Iterable[Basic], ground_truth_list: Iterable[Basic]) -> bool:
        """
        Check if any of the responses are correct and return true at first match
        :param response_list: list of responses
        :param ground_truth_list: list of ground truths
        :return: True if any response is correct
        """
        for answer in response_list:
            for ground_truth in ground_truth_list:
                try:
                    unsimplified_difference = answer - ground_truth
                    # check if the difference is close to zero with numpy
                    difference = self._safe_simplify_expression(unsimplified_difference)
                    tolerance = 1e-12
                    if abs(difference) < tolerance:
                        return True
                except ValueError:
                    # equations cannot be evaluated against each other
                    return False
        return False

    def _apply_safely(self, func: Callable[[Basic], Basic], list_of_expressions: list[Basic]) -> None:
        """
        apply safely to a list of expressions and replace the original expressions
        :param list_of_expressions: list of sympy expressions
        """
        for i, expression in enumerate(list_of_expressions):
            try:
                list_of_expressions[i] = func(expression)
            except RecursionError:
                list_of_expressions[i] = S.NaN

    def calculate(self, response: Completion) -> list[MetricResult]:
        """
        Calculate the accuracy of the completion

        performs several verification and simplification steps
        to ensure that the completion is correct

        the completion may either be a latex or string response
        which sympy will parse, factor, and simplify

        :param response: Completion object
        :return: list of MetricResult
        """
        ground_truths = []
        INVALID_ANSWER = S.NaN
        timeout = 10
        # latex parse all ingested ground truth values for math reasoning
        for gt in response.ground_truth_list:
            if gt is None:
                continue
            signal.signal(signal.SIGALRM, timeout_handler)  # Set timeout signal
            signal.alarm(timeout)  # Set timeout duration
            try:
                gt_normalized = self.normalize_expression(gt)
                gt_parsed = parse_latex(
                    gt_normalized
                )  # NOTE: parses f(x)=0,\quadf(x)=x-1,\quadf(x)=-x+1 to Eq(f(x), 0) ONLY
                ground_truths.append(gt_parsed)
            except Exception:
                ground_truths.append(gt)
            finally:
                # Ensure we never leak a pending alarm into later code paths.
                signal.alarm(0)
        normalized_response = self.normalize_expression(response.completion)
        response_list = self.check_for_equation(normalized_response)
        try:
            symb_is_correct = self._is_symbolically_equiv(response_list, ground_truths, INVALID_ANSWER)
        except Exception:
            symb_is_correct = False

        # check if already correct symbolically
        if symb_is_correct:
            return [
                MetricResult(
                    metric_name=self.NAME, value=float(symb_is_correct), higher_is_better=True, error=response.error
                )
            ]
        else:
            normalized_ground_truths = [
                self.normalize_expression(gt) for gt in response.ground_truth_list if gt is not None
            ]
            res = self._any_str_correct([normalized_response], normalized_ground_truths)
            return [MetricResult(metric_name=self.NAME, value=float(res), higher_is_better=True, error=response.error)]

    def _any_str_correct(self, response_list: list, ground_truths: list) -> bool:
        """
        Check if any of the responses are correct and return true at first match
        :param response_list: list of responses
        :param ground_truths: list of ground truths
        :return: True if any response is correct
        """
        for response in response_list:
            for ground_truth in ground_truths:
                if self._is_str_correct(response, ground_truth):
                    return True
        return False

    def _is_str_correct(self, str1: str, str2: str) -> bool:
        """
        Check if two strings are equal after stripping
        :param str1: first string
        :param str2: second string
        :param verbose: print the stripped strings
        :return: True if the strings are equal
        """
        # if multiple equal signs in ground truth (str2)
        # slide the response (str1) over the ground truth (str2)
        # at the interval of every equal sign in the ground truth
        # and check if any of the responses match
        # this accounts for generations such as b = 1 with ground truth as x = b = 1
        if str1.count("=") < str2.count("="):
            return self._is_str_correct(str1, str2[str2.index("=") + 1 :])
        if str1.count("=") > str2.count("="):
            return self._is_str_correct(str1[str1.index("=") + 1 :], str2)
        if str1 is None and str2 is None:
            return True
        if str1 is None or str2 is None:
            return False
        try:
            return str1 == str2
        except Exception:
            return str1 == str2

    def _is_symbolically_equiv(
        self, response_list: list[str], ground_truths: list, default_invalid: Basic = S.NaN
    ) -> bool:
        """
        Check if any of the responses are correct and return true at first match
        :param response_list: list of responses
        :param ground_truths: list of ground truths
        :param default_invalid: default value for invalid expressions
        :return: True if any response
        """

        try:
            self._apply_safely(parse_latex, response_list)
        except (LaTeXParsingError, SympifyError, TypeError):
            response_list = [default_invalid]  # this can not occur as an answer.
            return False

        # map objects dont catch errors, so we use safe apply here
        self._apply_safely(self._safe_simplify_expression, ground_truths)
        self._apply_safely(self._safe_simplify_expression, response_list)

        # check if any of the simplified responses match any of the simplified ground truths
        try:
            is_correct = self._any_symb_correct(response_list, ground_truths)
            return is_correct
        except ValueError:
            return False
