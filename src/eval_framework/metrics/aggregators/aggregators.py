from typing import Protocol

import numpy as np
import pandas as pd
from scipy.special import comb


class Aggregator(Protocol):
    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs) -> pd.DataFrame:
        pass


def closed_form_passatk(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k, exact=False) / comb(n, k, exact=False)


class PassAtK(Aggregator):
    def __init__(self, k: int = 1):
        self.k = k
        self.name = f"Pass@{k}"

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs) -> pd.DataFrame:
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
    def __init__(self):
        self.name = "Macro-Averaging"

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs) -> pd.DataFrame:
        agg_dict = {
            "value": "mean",
        }
        other_cols = [c for c in response_df.columns if c not in identifier_columns and c != "value"]
        agg_dict.update({c: "first" for c in other_cols})
        return response_df.groupby(identifier_columns).agg(agg_dict).reset_index()


class Identity(Aggregator):
    def __init__(self):
        self.name = "Identity"

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs) -> pd.DataFrame:
        return response_df
