"""DROP F1 and exact match evaluation.

Logic adapted from AllenNLP DROP evaluation:
https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py
"""

import re
import string

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the linear sum assignment problem (minimize cost) using the Hungarian algorithm.

    Pure NumPy implementation. Returns (row_ind, col_ind) with row_ind sorted,
    matching scipy.optimize.linear_sum_assignment API for rectangular matrices.
    """
    cost = np.atleast_2d(np.asarray(cost_matrix, dtype=np.float64))
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    # Pad to square with zeros so unassigned rows/cols contribute zero cost
    C = np.zeros((n, n))
    C[:n_rows, :n_cols] = cost

    # Row and column reductions
    u = np.min(C, axis=1)
    u[u == np.inf] = 0
    C = C - u[:, np.newaxis]
    v = np.min(C, axis=0)
    v[v == np.inf] = 0
    C = C - v[np.newaxis, :]

    # Starred zeros: assignment (1 = starred). Start with no stars.
    star = np.zeros((n, n), dtype=np.intp)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)
    prime = np.zeros((n, n), dtype=np.intp)

    def find_zero() -> tuple[int, int] | None:
        for i in range(n):
            if row_covered[i]:
                continue
            for j in range(n):
                if not col_covered[j] and C[i, j] == 0:
                    return (i, j)
        return None

    def star_in_row(i: int) -> int | None:
        for j in range(n):
            if star[i, j]:
                return j
        return None

    def star_in_col(j: int) -> int | None:
        for i in range(n):
            if star[i, j]:
                return i
        return None

    def prime_in_row(i: int) -> int | None:
        for j in range(n):
            if prime[i, j]:
                return j
        return None

    while True:
        # Cover columns containing a starred zero
        col_covered[:] = False
        for j in range(n):
            for i in range(n):
                if star[i, j]:
                    col_covered[j] = True
                    break
        if np.all(col_covered):
            break
        prime[:] = 0
        row_covered[:] = False
        while True:
            z = find_zero()
            if z is None:
                # No uncovered zero: find minimum uncovered value and adjust
                min_val = np.inf
                for i in range(n):
                    if not row_covered[i]:
                        for j in range(n):
                            if not col_covered[j] and C[i, j] < min_val:
                                min_val = C[i, j]
                if min_val == np.inf or min_val <= 0:
                    min_val = 1e-10
                for i in range(n):
                    if row_covered[i]:
                        C[i, :] += min_val
                for j in range(n):
                    if not col_covered[j]:
                        C[:, j] -= min_val
                continue
            i, j = z
            prime[i, j] = 1
            cj = star_in_row(i)
            if cj is None:
                # Augmenting path: unstar starred, star primed along path
                path = [(i, j)]
                while True:
                    ji = star_in_col(path[-1][1])
                    if ji is None:
                        break
                    path.append((ji, path[-1][1]))
                    pj = prime_in_row(ji)
                    if pj is None:
                        break
                    path.append((ji, pj))
                for (pi, pj) in path:
                    star[pi, pj] = 1 - star[pi, pj]
                prime[:] = 0
                row_covered[:] = False
                col_covered[:] = False
                break
            row_covered[i] = True
            col_covered[cj] = False

    # Extract assignment: (row_ind, col_ind) for starred zeros, row_ind sorted
    row_ind = np.array([i for i in range(n) for j in range(n) if star[i, j]], dtype=np.intp)
    col_ind = np.array([j for i in range(n) for j in range(n) if star[i, j]], dtype=np.intp)
    # Keep only assignments within original matrix
    mask = (row_ind < n_rows) & (col_ind < n_cols)
    row_ind = row_ind[mask]
    col_ind = col_ind[mask]
    # Sort by row index (scipy API)
    perm = np.argsort(row_ind)
    return row_ind[perm], col_ind[perm]


def process_results(doc: dict, results: list) -> dict[str, float]:
    """Compute DROP exact_match and F1 between predictions and gold answers.

    doc["answers"] is a list of gold answers (each a tuple or list of strings).
    results is a list of predicted answers (one per prediction; for one completion use [pred]).
    """
    preds, golds = results, doc["answers"]
    max_em = 0.0
    max_f1 = 0.0
    for gold_answer in golds:
        exact_match, f1_score = get_metrics(preds, gold_answer)
        if gold_answer and (gold_answer[0].strip() if isinstance(gold_answer[0], str) else True):
            max_em = max(max_em, exact_match)
            max_f1 = max(max_f1, f1_score)
    return {"exact_match": max_em, "f1": max_f1}


def get_metrics(predicted: list | str, gold: tuple | list) -> tuple[float, float]:
    """Return (exact_match, f1) for one gold answer. predicted and gold can be string or list of strings."""
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = float(np.mean(f1_per_bag))
    f1 = round(f1, 2)
    return exact_match, f1


def _answer_to_bags(answer: list | tuple | str) -> tuple[list[str], list[set]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = list(answer)
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize(str(raw_span))
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: list[set], gold: list[set]) -> np.ndarray:
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = _linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: set, gold_bag: set) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    return (2 * precision * recall) / (precision + recall) if (precision or recall) else 0.0


def _match_numbers_if_present(gold_bag: set, predicted_bag: set) -> bool:
    gold_numbers = {w for w in gold_bag if _is_number(w)}
    predicted_numbers = {w for w in predicted_bag if _is_number(w)}
    return (not gold_numbers) or bool(gold_numbers.intersection(predicted_numbers))


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text: str) -> str:
    return _ARTICLES.sub(" ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude) if not _is_number(text) else text


def _fix_number(text: str) -> str:
    return str(float(text)) if _is_number(text) else text


def _tokenize(text: str) -> list[str]:
    return re.split(" |-", text)


def _normalize(answer: str) -> str:
    tokens = [
        _white_space_fix(_remove_articles(_fix_number(_remove_punc(token.lower())))) for token in _tokenize(answer)
    ]
    tokens = [t for t in tokens if t.strip()]
    return " ".join(tokens).strip()
