import re
import string
from typing import Any

import numpy as np

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class AccuracyCompletion(BaseMetric[Completion]):
    NAME = "Accuracy Completion"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truths = response.ground_truth_list
        is_correct = any(response.completion == gt for gt in ground_truths)
        return [
            MetricResult(metric_name=self.NAME, value=float(is_correct), higher_is_better=True, error=response.error)
        ]


class AccuracyCompletionWithEvaluate(AccuracyCompletion):
    def __init__(self, regexes_to_ignore: list[str], ignore_case: bool = False, ignore_punctuation: bool = False):
        self.regexes_to_ignore = regexes_to_ignore
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]
        ground_truths = response.ground_truth_list
        model_answer = response.completion

        is_correct = exact_match_hf_evaluate(
            predictions=[model_answer] * len(ground_truths),
            references=ground_truths,  # type: ignore[arg-type]
            regexes_to_ignore=self.regexes_to_ignore,
            ignore_case=self.ignore_case,
            ignore_punctuation=self.ignore_punctuation,
        )["exact_match"]
        return [
            MetricResult(metric_name=self.NAME, value=float(is_correct), higher_is_better=True, error=response.error)
        ]


class AccuracyCompletionOLMES(AccuracyCompletionWithEvaluate):
    # If we did a functools partial, code fails as there an issubclass check that
    # doesn't work with partial. These specific regexes are taken from
    # https://github.com/allenai/olmes/blob/main/oe_eval/tasks/oe_eval_tasks/gsm8k.py#L70
    def __init__(self) -> None:
        super().__init__(regexes_to_ignore=[",", "\\$", "(?s).*#### ", "\\.$"])


# The following code is (largely) reproduced from https://github.com/allenai/olmes/blob/main/oe_eval/dependencies/hf_evaluate/exact_match.py#L25
# Olmes released under Apache 2.0 license and so is the HF evaluate library.
# Some cosmetic modifications have been made to fit our codebase and linting rules.
# -------------------------------------------------------------------------------------

### Code ported from Huggingface's `evaluate` library at
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.
### Port taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/metrics.py used
### to fix the issue: https://github.com/EleutherAI/lm-evaluation-harness/pull/2045

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions: list[str],
    references: list[str],
    regexes_to_ignore: list[str] | None = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> dict[str, Any]:  # type: ignore
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])  # type: ignore
            references = np.array([re.sub(s, "", x) for x in references])  # type: ignore
    else:
        predictions = np.asarray(predictions)  # type: ignore
        references = np.asarray(references)  # type: ignore

    if ignore_case:
        predictions = np.char.lower(predictions)  # type: ignore
        references = np.char.lower(references)  # type: ignore

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)  # type: ignore
        references = np.char.translate(references, table=repl_table)  # type: ignore

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)  # type: ignore
        references = np.char.translate(references, table=repl_table)  # type: ignore

    # NOTE: For multiple ground-truths OLMES returns the mean over their scores. The max over
    # it would be more meaningful, but we leave it here for parity.
    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}
