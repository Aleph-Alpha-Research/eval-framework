from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.special import comb


class Aggregator(Protocol):
    """Base class for metric aggregators.
    
    An aggregator collapses multiple evaluation rows for the same problem (i.e. prompt) into a
    single score per problem. The input DataFrame has one row per (problem, attempt)
    pair; the output has one row per problem with a new ``value``.
    
    Args:
        response_df: DataFrame where each row is one evaluation attempt. Must contain
            a ``value`` column (the per-attempt score) and all ``identifier_columns``.
        identifier_columns: Columns that uniquely identify a problem (e.g. ``["prompt"]``).
            Rows sharing the same identifier are different attempts at the same problem.
            
    Returns:
        DataFrame with one row per unique problem and a ``value`` column holding
        the aggregated score. All non-identifier, non-value columns are preserved
        (typically via ``"first"``).
        
    Example input (``identifier_columns=["prompt"]``, 3 attempts per problem):
    
        | prompt         | value | subject |
        |----------------|-------|---------|
        | "What is 2+2?" |  1.0  | algebra |
        | "What is 2+2?" |  1.0  | algebra |
        | "What is 2+2?" |  0.0  | algebra |
        | "Solve x^2=4"  |  0.0  | algebra |
        | "Solve x^2=4"  |  1.0  | algebra |
        | "Solve x^2=4"  |  0.0  | algebra |
    """
    
    name: str

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs: Any) -> pd.DataFrame: ...


def closed_form_passatk(n: int, c: int, k: int) -> float:
    """Closed-form pass@k estimator (see HumanEval paper).

    pass@k = 1 - C(n-c, k) / C(n, k)

    Given n total samples with c correct, this is the probability that at least one of k
    randomly chosen samples is correct. The ratio C(n-c,k)/C(n,k) is the chance all k picks
    are wrong; subtracting from 1 gives success probability. When n-c < k there aren't enough
    wrong samples to fill k slots, so the result is trivially 1.
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k, exact=False) / comb(n, k, exact=False)


class PassAtK(Aggregator):
    """Computes pass@k: the probability that at least one of k random attempts is correct.
    
    Groups rows by ``identifier_columns``, counts correct (``c = sum(value)``) and
    total (``n = count(value)``) attempts per problem, then applies the closed-form
    estimator.
    
    Expects ``value`` to be binary (0 or 1). For k=1 this is equivalent to the mean.
    
    Example (k=2, continuing from the Aggregator docstring example):
        "What is 2+2?": n=3, c=2, k=2 -> 1.0  (guaranteed correct pick)
        "Solve x^2=4":  n=3, c=1,  k=2 -> 0.667 (as computed by the `closed_form_passatk`)
        
        Output:
        | prompt         | value | subject |
        |----------------|-------|---------|
        | "What is 2+2?" | 1.000 | algebra |
        | "Solve x^2=4"  | 0.667 | algebra |
    """

    def __init__(self, k: int = 1):
        self.k = k
        self.name = f"Pass@{k}"

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs: Any) -> pd.DataFrame:
        # agg_dict decides how each column (`agg_dict` key) will get aggregated (`agg_dict` value).
        # For the `value` column, we compute both the sum and the count, for all other columns we simply pick the first entry (as they are identical anyway).
        other_cols = [c for c in response_df.columns if c not in identifier_columns and c != "value"]
        agg_dict = {"value": ["sum", "count"], **{c: "first" for c in other_cols}}
        agg = response_df.groupby(identifier_columns).agg(agg_dict)
        # flatten multi-index columns from value agg: ("value", "sum") / ("value", "count")
        c = agg[("value", "sum")].values
        n = agg[("value", "count")].values
        scores = np.array([closed_form_passatk(n_i, c_i, self.k) for n_i, c_i in zip(n, c)])
        out = agg.drop(columns=[("value", "sum"), ("value", "count")])
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.droplevel(1)
        return out.assign(value=scores).reset_index()


class IdentifierMean(Aggregator):
    """Computes the mean of the ``value`` column per problem group."""

    def __init__(self) -> None:
        self.name = "IdentifierMean"

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs: Any) -> pd.DataFrame:
        agg_dict = {
            "value": "mean",
        }
        other_cols = [c for c in response_df.columns if c not in identifier_columns and c != "value"]
        agg_dict.update({c: "first" for c in other_cols})
        return response_df.groupby(identifier_columns).agg(agg_dict).reset_index()


class Identity(Aggregator):
    def __init__(self) -> None:
        self.name = "Identity"

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs: Any) -> pd.DataFrame:
        return response_df
