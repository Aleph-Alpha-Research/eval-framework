"""Corpus-level BPB estimators.

Pure functions over parallel (bits, nbytes[, tokens]) arrays. Not per-item
BaseMetrics: each collapses a batch of (bits, nbytes) pairs to one scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

COMMON_BINS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 10**9)]
NSTAR_GRID = ("mean", "median", 16, 64, 256, "inf")


def _as_float(a: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected 1-d array, got shape {x.shape}")
    return x


def mean_of_ratios(bits: np.ndarray, nbytes: np.ndarray) -> float:
    n = _as_float(nbytes)
    if np.any(n <= 0):
        raise ValueError("nbytes must be positive")
    return float(np.mean(_as_float(bits) / n))


def corpus_bpb(bits: np.ndarray, nbytes: np.ndarray) -> float:
    n = _as_float(nbytes)
    if n.sum() <= 0:
        raise ValueError("sum(nbytes) must be positive")
    return float(_as_float(bits).sum() / n.sum())


def bits_per_answer(bits: np.ndarray) -> float:
    return float(np.mean(_as_float(bits)))


def space_stripped_corpus_bpb(bits: np.ndarray, nbytes: np.ndarray, leading_space: np.ndarray) -> float:
    """Corpus BPB after removing one leading-space byte where the gold had one."""
    n = _as_float(nbytes) - np.asarray(leading_space, dtype=float)
    n = np.maximum(n, 1.0)
    return corpus_bpb(bits, n)


@dataclass
class AffineFit:
    intercept: float
    slope: float
    r2: float
    method: str


def ols_affine(nbytes: np.ndarray, bits: np.ndarray) -> AffineFit:
    x = _as_float(nbytes)
    y = _as_float(bits)
    if len(x) < 2 or np.std(x) == 0:
        return AffineFit(float(np.mean(y)), 0.0, float("nan"), "ols")
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return AffineFit(float(intercept), float(slope), r2, "ols")


def huber_affine(nbytes: np.ndarray, bits: np.ndarray, c: float = 1.345, max_iter: int = 50) -> AffineFit:
    """IRLS Huber regression of bits on nbytes with intercept."""
    x = _as_float(nbytes)
    y = _as_float(bits)
    n = len(x)
    if n < 3 or np.std(x) == 0:
        return ols_affine(x, y)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(max_iter):
        resid = y - X @ beta
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = mad / 0.6745 if mad > 0 else (np.std(resid) or 1.0)
        u = resid / (c * scale)
        w = np.ones(n)
        mask = np.abs(u) > 1
        w[mask] = 1.0 / np.abs(u[mask])
        W = np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return AffineFit(float(beta[0]), float(beta[1]), r2, "huber")


def ols_leverage_weights(nbytes: np.ndarray, n_star: float) -> np.ndarray:
    """h_i such that predicted L(n*) = sum h_i L_i."""
    x = _as_float(nbytes)
    n = len(x)
    xbar = float(x.mean())
    ss = float(np.sum((x - xbar) ** 2))
    if ss == 0:
        return np.full(n, 1.0 / n)
    return (1.0 / n) + (n_star - xbar) * (x - xbar) / ss


def bpb_at_nstar(bits: np.ndarray, nbytes: np.ndarray, n_star: float, method: str = "ols") -> dict[str, Any]:
    x = _as_float(nbytes)
    y = _as_float(bits)
    fit = huber_affine(x, y) if method == "huber" else ols_affine(x, y)
    if not np.isfinite(n_star) or n_star == 0:
        value = fit.slope
        in_support = False
    else:
        value = (fit.intercept + fit.slope * n_star) / n_star
        p5, p95 = np.percentile(x, [5, 95])
        in_support = bool(p5 <= n_star <= p95)
    h = ols_leverage_weights(x, n_star if np.isfinite(n_star) else x.mean())
    return {
        "value": float(value),
        "alpha": fit.intercept,
        "beta": fit.slope,
        "r2": fit.r2,
        "method": method,
        "n_star": n_star,
        "in_support": in_support,
        "frac_negative_weights": float(np.mean(h < 0)) if np.isfinite(n_star) else float("nan"),
        "max_abs_weight": float(np.max(np.abs(h))),
        "n_bar": float(x.mean()),
    }


def resolve_nstar(nbytes: np.ndarray, spec: float | int | str) -> float:
    x = _as_float(nbytes)
    if spec == "mean":
        return float(x.mean())
    if spec == "median":
        return float(np.median(x))
    if spec == "inf":
        return float("inf")
    return float(spec)


def effective_length_bpb(bits: np.ndarray, nbytes: np.ndarray) -> dict[str, Any]:
    """mean L_i / (n_i + alpha/beta). Undefined when beta <= 0."""
    fit = ols_affine(nbytes, bits)
    if fit.slope <= 1e-12:
        return {"value": float("nan"), "defined": False, "alpha": fit.intercept, "beta": fit.slope}
    offset = fit.intercept / fit.slope
    denom = _as_float(nbytes) + offset
    if np.any(denom <= 0):
        return {"value": float("nan"), "defined": False, "alpha": fit.intercept, "beta": fit.slope}
    return {
        "value": float(np.mean(_as_float(bits) / denom)),
        "defined": True,
        "alpha": fit.intercept,
        "beta": fit.slope,
        "offset": offset,
    }


def _bin_index(n: float, edges: list[tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(edges):
        if lo <= n <= hi:
            return i
    return len(edges) - 1


def _assign_bins(arr: np.ndarray, edges: list[tuple[float, float]]) -> np.ndarray:
    """Bin index per element; left-closed right-open except the last bin, which is closed.

    Values outside the outer edges are clamped into the first or last bin.
    """
    idx = np.empty(len(arr), dtype=int)
    for i, v in enumerate(arr):
        placed = False
        for b, (lo, hi) in enumerate(edges):
            last = b == len(edges) - 1
            if (lo <= v <= hi) if last else (lo <= v < hi):
                idx[i] = b
                placed = True
                break
        if not placed:
            idx[i] = 0 if v < edges[0][0] else len(edges) - 1
    return idx


def bin_mass(nbytes: np.ndarray, bins: list[tuple[int, int]]) -> np.ndarray:
    """Fraction of `nbytes` falling in each bin. Sums to 1."""
    n = _as_float(nbytes)
    edges = [(float(lo), float(hi)) for lo, hi in bins]
    idx = _assign_bins(n, edges)
    return np.array([(idx == b).mean() for b in range(len(edges))], dtype=float)


def ls_bpb(
    bits: np.ndarray,
    nbytes: np.ndarray,
    q_nbytes: np.ndarray | None = None,
    bins: list[tuple[int, int]] | None = None,
    n_quintiles: int = 5,
    q_mass: np.ndarray | None = None,
) -> dict[str, Any]:
    """Length-standardized BPB with fixed nonnegative stratum weights.

    q_nbytes: lengths that define q (default: nbytes). bins=None uses quintiles of q_nbytes.
    q_mass: explicit stratum weights instead of a q_nbytes histogram (e.g. equal task weight).
    """
    y = _as_float(bits)
    n = _as_float(nbytes)
    qn = _as_float(q_nbytes) if q_nbytes is not None else n
    if bins is None:
        qs = np.quantile(qn, np.linspace(0, 1, n_quintiles + 1))
        qs[0] = min(qs[0], n.min(), qn.min())
        qs[-1] = max(qs[-1], n.max(), qn.max())
        # left-closed right-open except last
        edges = [(float(qs[i]), float(qs[i + 1])) for i in range(len(qs) - 1)]
    else:
        edges = [(float(lo), float(hi)) for lo, hi in bins]

    y_idx = _assign_bins(n, edges)
    k = len(edges)
    if q_mass is not None:
        q_mass = _as_float(q_mass)
        if len(q_mass) != k:
            raise ValueError(f"q_mass has {len(q_mass)} strata but there are {k} bins")
        if (q_mass < 0).any():
            raise ValueError("q_mass must be nonnegative")
        total = q_mass.sum()
        if total <= 0:
            raise ValueError("q_mass must have positive total mass")
        q_mass = q_mass / total
    else:
        q_idx = _assign_bins(qn, edges)
        q_mass = np.array([(q_idx == b).mean() for b in range(k)], dtype=float)
    mean_L = np.full(k, np.nan)
    mean_n = np.full(k, np.nan)
    counts = np.zeros(k, dtype=int)
    for b in range(k):
        m = y_idx == b
        counts[b] = int(m.sum())
        if m.any():
            mean_L[b] = y[m].mean()
            mean_n[b] = n[m].mean()
    supported = (q_mass > 0) & (counts > 0)
    if not supported.any():
        return {"value": float("nan"), "support_mismatch": True, "q_mass": q_mass.tolist(), "counts": counts.tolist()}
    w = q_mass.copy()
    w[~supported] = 0.0
    w = w / w.sum()
    value = float((w[supported] * mean_L[supported]).sum() / (w[supported] * mean_n[supported]).sum())
    return {
        "value": value,
        "support_mismatch": bool((q_mass > 0).sum() != supported.sum()),
        "weights": w.tolist(),
        "q_mass": q_mass.tolist(),
        "counts": counts.tolist(),
        "mean_L": mean_L.tolist(),
        "mean_n": mean_n.tolist(),
        "n_negative_weights": 0,
    }


# Prefix BPB: score the first K content bytes of each gold (fixed K for all items).
# offset=1 drops one leading space from the content-byte count; its bits stay in the numerator.


@dataclass
class PrefixItem:
    """Byte-indexed costs for one item (prefix / Prior BPB).

    bits[j]: -log2 prob of token j; byte_lens[j]: UTF-8 length of token j;
    offset: content bytes to skip at the start (0 or 1 for a leading space).
    """

    bits: np.ndarray
    byte_lens: np.ndarray
    offset: int = 0

    def __post_init__(self) -> None:
        self.bits = _as_float(self.bits)
        self.byte_lens = _as_float(self.byte_lens)
        if len(self.bits) != len(self.byte_lens):
            raise ValueError("bits and byte_lens must have equal length")
        if np.any(self.byte_lens < 0):
            raise ValueError("byte_lens must be nonnegative")

    @property
    def total_bytes(self) -> int:
        return int(round(float(self.byte_lens.sum())))

    @property
    def content_bytes(self) -> int:
        return self.total_bytes - int(self.offset)


def make_prefix_items(
    per_token: list[tuple[list[float] | np.ndarray, list[float] | np.ndarray]],
    choices: list[str] | None = None,
) -> list[PrefixItem]:
    """Build PrefixItems from parallel (bits, byte_lens) pairs.

    If choices is given, a leading space sets offset=1 (excluded from content bytes).
    """
    items: list[PrefixItem] = []
    for i, (b, m) in enumerate(per_token):
        offset = 1 if (choices is not None and choices[i].startswith(" ")) else 0
        items.append(PrefixItem(np.asarray(b, dtype=float), np.asarray(m, dtype=float), offset=offset))
    return items


def cumulative_cost(bits: np.ndarray, byte_lens: np.ndarray, k_abs: float) -> float:
    """Bits for the first k_abs bytes. Linear split inside a token that straddles k."""
    b = _as_float(bits)
    m = _as_float(byte_lens)
    if len(b) == 0:
        return 0.0
    B = np.cumsum(m)
    S = np.cumsum(b)
    n = float(B[-1])
    if k_abs <= 0:
        return 0.0
    if k_abs >= n:
        return float(S[-1])
    # j = first token whose cumulative bytes exceed k_abs
    j = int(np.searchsorted(B, k_abs, side="right"))
    prev_cost = float(S[j - 1]) if j > 0 else 0.0
    prev_bytes = float(B[j - 1]) if j > 0 else 0.0
    mj = float(m[j])
    if mj <= 0:
        return prev_cost
    return prev_cost + float(b[j]) * (k_abs - prev_bytes) / mj


def _item_cost(item: PrefixItem, k_content: float) -> float:
    """Bits to cover the first `k_content` content bytes (skips the offset)."""
    return cumulative_cost(item.bits, item.byte_lens, float(item.offset) + k_content)


def _supported(items: list[PrefixItem], k: float) -> list[PrefixItem]:
    return [it for it in items if it.content_bytes >= k]


def pbpb(items: list[PrefixItem], k: float) -> float:
    """Prefix BPB at budget k: mean_i C_i(k)/k over items with content_bytes >= k."""
    if k <= 0:
        raise ValueError("k must be positive")
    sup = _supported(items, k)
    if not sup:
        return float("nan")
    return float(np.mean([_item_cost(it, k) for it in sup]) / k)


def entry_cost(items: list[PrefixItem], k0: float) -> float:
    """A(k0): mean bits to cover the first k0 content bytes, over items with support."""
    sup = _supported(items, k0)
    if not sup:
        return float("nan")
    return float(np.mean([_item_cost(it, k0) for it in sup]))


def continuation_rate(items: list[PrefixItem], k1: float, k2: float) -> float:
    """R(k1,k2): mean per-byte bits from content byte k1 to k2, over items with support."""
    if k2 <= k1:
        raise ValueError("k2 must exceed k1")
    sup = _supported(items, k2)
    if not sup:
        return float("nan")
    deltas = [(_item_cost(it, k2) - _item_cost(it, k1)) / (k2 - k1) for it in sup]
    return float(np.mean(deltas))


def prefix_decompose(items: list[PrefixItem], k0: float, k: float) -> dict[str, Any]:
    """Exact split PBPB(k) = [A(k0) + R(k0,k)*(k-k0)] / k on a common support (>= k)."""
    if not (0 < k0 < k):
        raise ValueError("require 0 < k0 < k")
    sup = _supported(items, k)
    n = len(sup)
    if n == 0:
        return {"pbpb": float("nan"), "A": float("nan"), "R": float("nan"), "n_items": 0, "coverage": 0.0}
    a = float(np.mean([_item_cost(it, k0) for it in sup]))
    r = float(np.mean([(_item_cost(it, k) - _item_cost(it, k0)) / (k - k0) for it in sup]))
    value = (a + r * (k - k0)) / k
    return {
        "pbpb": value,
        "A": a,
        "R": r,
        "k0": k0,
        "k": k,
        "n_items": n,
        "coverage": n / len(items) if items else 0.0,
    }


def prefix_curve(items: list[PrefixItem], ks: list[float] | np.ndarray) -> list[dict[str, Any]]:
    """PBPB and coverage across a grid of byte budgets `ks`."""
    n_total = len(items)
    out: list[dict[str, Any]] = []
    for k in ks:
        sup = _supported(items, float(k))
        cov = (len(sup) / n_total) if n_total else 0.0
        value = float(np.mean([_item_cost(it, float(k)) for it in sup]) / float(k)) if sup else float("nan")
        out.append({"k": float(k), "value": value, "coverage": cov, "n_items": len(sup)})
    return out


# Prior BPB: PriorBPB = sum_k s(k) R(k) / sum_k s(k), where R(k) is
# the mean marginal cost at content byte k and s(k) comes from a fixed length prior.
# Task empirical lengths as prior recover corpus BPB; geometric prior with mean mu and
# affine C(k)=alpha+beta*k gives alpha/mu + beta.


def per_byte_cost(item: PrefixItem) -> np.ndarray:
    """Bits per content byte. Same linear split as cumulative_cost; zero-byte tokens
    attach to the previous byte; offset bytes are skipped."""
    bits = item.bits
    m = item.byte_lens
    per: list[float] = []
    carry = 0.0
    for bj, mj_f in zip(bits, m):
        mj = int(round(float(mj_f)))
        if mj <= 0:
            if per:
                per[-1] += float(bj)
            else:
                carry += float(bj)
            continue
        share = float(bj) / mj
        per.append(share + carry)
        carry = 0.0
        per.extend([share] * (mj - 1))
    if carry and per:
        per[-1] += carry
    arr = np.asarray(per, dtype=float)
    return arr[item.offset :]


def marginal_rate_curve(items: list[PrefixItem], kmax: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(k, R(k), coverage(k)) for k=1..kmax, using every byte of every item.

    R(k) is the mean marginal cost at content byte k over all items that reach it.
    """
    lengths = [it.content_bytes for it in items]
    if not lengths:
        return np.array([]), np.array([]), np.array([])
    top = max(lengths)
    kmax = top if kmax is None else min(kmax, top)
    tot = np.zeros(kmax)
    cnt = np.zeros(kmax)
    for it in items:
        pbc = per_byte_cost(it)
        kk = min(len(pbc), kmax)
        if kk <= 0:
            continue
        tot[:kk] += pbc[:kk]
        cnt[:kk] += 1
    R = np.divide(tot, cnt, out=np.full(kmax, np.nan), where=cnt > 0)
    coverage = cnt / len(items)
    return np.arange(1, kmax + 1), R, coverage


def mean_cost_curve(items: list[PrefixItem], kmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean cumulative cost A(k) = sum_{j<=k} R(j)."""
    k, R, _ = marginal_rate_curve(items, kmax)
    if len(k) == 0:
        return k, R
    return k, np.cumsum(np.nan_to_num(R))


def geometric_survival(mu: float, kmax: int) -> np.ndarray:
    """s(k) = (1 - 1/mu)^(k-1), k=1..kmax. Mean-mu geometric length prior."""
    if mu <= 1:
        raise ValueError("prior mean mu must exceed 1")
    p = 1.0 / mu
    k = np.arange(kmax)
    return np.power(1.0 - p, k)


def reference_survival(ref_lengths: np.ndarray, kmax: int) -> np.ndarray:
    """s(k) = P(N >= k) for a fixed reference length sample, k=1..kmax."""
    ref = _as_float(ref_lengths)
    return np.array([float(np.mean(ref >= k)) for k in range(1, kmax + 1)], dtype=float)


def prior_bpb(
    items: list[PrefixItem],
    prior: str = "geometric",
    mu: float | None = None,
    ref_lengths: np.ndarray | None = None,
    kmax: int | None = None,
) -> dict[str, Any]:
    """Prior BPB: weighted average of R(k) under a fixed length prior.

    prior="geometric": mean mu. prior="reference": ref_lengths (task lengths give corpus BPB).
    """
    k, R, coverage = marginal_rate_curve(items, kmax)
    if len(k) == 0:
        return {"value": float("nan"), "prior": prior, "n_items": 0}
    kk = len(k)
    if prior == "geometric":
        if mu is None:
            raise ValueError("geometric prior needs mu")
        s = geometric_survival(mu, kk)
    elif prior == "reference":
        if ref_lengths is None:
            raise ValueError("reference prior needs ref_lengths")
        s = reference_survival(ref_lengths, kk)
    else:
        raise ValueError(f"unknown prior {prior!r}")
    Rf = np.nan_to_num(R)
    denom = float(s.sum())
    value = float((s * Rf).sum() / denom) if denom > 0 else float("nan")
    return {
        "value": value,
        "prior": prior,
        "mu": mu if prior == "geometric" else float(np.average(np.arange(1, kk + 1), weights=s)),
        "prior_mass_in_support": float((s * (coverage > 0)).sum() / denom) if denom > 0 else 0.0,
        "kmax": kk,
        "n_items": len(items),
    }


def summarize_all(
    bits: np.ndarray,
    nbytes: np.ndarray,
    tokens: np.ndarray | None = None,
    q_nbytes: np.ndarray | None = None,
    leading_space: np.ndarray | None = None,
) -> dict[str, Any]:
    bits = _as_float(bits)
    nbytes = _as_float(nbytes)
    out: dict[str, Any] = {
        "n_items": int(len(bits)),
        "mean_of_ratios": mean_of_ratios(bits, nbytes),
        "corpus_bpb": corpus_bpb(bits, nbytes),
        "bits_per_answer": bits_per_answer(bits),
        "mean_bytes": float(nbytes.mean()),
        "median_bytes": float(np.median(nbytes)),
        "ols": ols_affine(nbytes, bits).__dict__,
        "huber": huber_affine(nbytes, bits).__dict__,
        "ls_bpb_task_q": ls_bpb(bits, nbytes, q_nbytes=q_nbytes),
        "ls_bpb_common_q": ls_bpb(bits, nbytes, q_nbytes=q_nbytes, bins=COMMON_BINS),
        "effective_length": effective_length_bpb(bits, nbytes),
        "bpb_at_nstar_ols": {},
        "bpb_at_nstar_huber": {},
    }
    if tokens is not None:
        tok = _as_float(tokens)
        if np.all(np.isfinite(tok)) and tok.sum() > 0:
            out["token_corpus_bpb"] = corpus_bpb(bits, tok)
            out["ols_tokens"] = ols_affine(tok, bits).__dict__
    if leading_space is not None:
        out["space_stripped_corpus_bpb"] = space_stripped_corpus_bpb(bits, nbytes, leading_space)
    for spec in NSTAR_GRID:
        nstar = resolve_nstar(nbytes if q_nbytes is None else q_nbytes, spec)
        key = str(spec)
        out["bpb_at_nstar_ols"][key] = bpb_at_nstar(bits, nbytes, nstar, "ols")
        out["bpb_at_nstar_huber"][key] = bpb_at_nstar(bits, nbytes, nstar, "huber")
    return out
