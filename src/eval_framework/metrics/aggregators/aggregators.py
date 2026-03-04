from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.special import comb


class Aggregator(Protocol):
    name: str

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs: Any) -> pd.DataFrame:
        pass


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
    """Computes pass@k per problem group.

    Each row is a (problem, sample) pair with a binary ``value`` (1=correct). The groupby
    aggregates ``value`` with both ``sum`` (-> c) and ``count`` (-> n), while keeping the
    ``first`` of every other column. Because ``value`` gets two agg functions, pandas
    creates a MultiIndex on columns — e.g. ("value","sum"), ("other_cols","first"). After
    computing scores and dropping the value tuples, ``droplevel(1)`` flattens the leftover
    metadata MultiIndex back to plain column names. The result is one row per problem with
    the pass@k score as ``value``.
    """

    def __init__(self, k: int = 1):
        self.k = k
        self.name = f"Pass@{k}"

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs: Any) -> pd.DataFrame:
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
        self.name = "Macro-Averaging"

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
