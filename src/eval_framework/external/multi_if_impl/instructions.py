# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: ignore-errors

"""Multi-IF's language-aware variants of the IFEval checkers, and the registry that swaps them in."""

import re
from collections.abc import Iterable

import langdetect
from langdetect import detector_factory

from eval_framework.external.ifeval_impl import instructions, instructions_registry, instructions_util

_CHINESE_CHARS_PATTERN = r"[\u4E00-\u9FFF\u3400-\u4DBF]"
_JAPANESE_CHARS_PATTERN = r"[\u3040-\u309f\u30a0-\u30ff]"
_KOREAN_CHARS_PATTERN = r"[\uAC00-\uD7AF]"
_CHINESE_JAPANESE_CODES = ("zh", "zh-cn", "zh-tw", "ja")


def _detect_language(text):
    # langdetect samples randomly and is unseeded by default, so short texts can flip language between calls.
    # Seeding this detector only keeps grading reproducible without changing langdetect's global behaviour.
    detector_factory.init_factory()
    detector = detector_factory._factory.create()
    detector.seed = 0
    try:
        detector.append(text)
        return detector.detect()
    except langdetect.LangDetectException:
        return "en"


def split_chinese_japanese(lines: str) -> Iterable[str]:
    """Split Chinese and Japanese text into sentences, line by line."""
    for line in lines.splitlines():
        yield from re.findall(r"[^!?。\.\!\?\！\？\．\n]+[!?。\.\!\?\！\？\．\n]?", line.strip(), flags=re.U)


def count_words_chinese_japanese(text: str) -> int:
    """Count each CJK character as a word and whitespace-separated runs of anything else as one word."""
    non_alphanumeric_patterns = (
        r"[\\.\!\?\．\/_,\{\}<>:;$%^&*(+\"\'+——！，。？、`~@#￥……（）：；《）《》“”()\[\]»〔〕\-「」]+"
    )
    text = re.sub(non_alphanumeric_patterns, "", text)
    foreign_chars_patterns = "|".join([_CHINESE_CHARS_PATTERN, _JAPANESE_CHARS_PATTERN, _KOREAN_CHARS_PATTERN])
    asian_chars_cnt = len(re.findall(foreign_chars_patterns, text))
    non_asian_words_cnt = len(re.sub(foreign_chars_patterns, " ", text).split())
    return non_asian_words_cnt + asian_chars_cnt


def count_hindi_num_sentences(text):
    sentences = re.split(r"(?<=[।!?])\s*", text)
    return len([s for s in sentences if s.strip()])


def _compare(count, relation, threshold):
    if relation == instructions._COMPARISON_RELATION[0]:
        return count < threshold
    elif relation == instructions._COMPARISON_RELATION[1]:
        return count >= threshold


class NumberOfSentences(instructions.NumberOfSentences):
    def check_following(self, value):
        lang = _detect_language(value)
        if lang == "hi":
            num_sentences = count_hindi_num_sentences(value)
        elif lang in _CHINESE_JAPANESE_CODES:
            num_sentences = len(list(split_chinese_japanese(value)))
        else:
            num_sentences = instructions_util.count_sentences(value)
        return _compare(num_sentences, self._comparison_relation, self._num_sentences_threshold)


class NumberOfWords(instructions.NumberOfWords):
    def check_following(self, value):
        if _detect_language(value) in _CHINESE_JAPANESE_CODES:
            num_words = count_words_chinese_japanese(value)
        else:
            num_words = instructions_util.count_words(value)
        return _compare(num_words, self._comparison_relation, self._num_words)


class QuotationChecker(instructions.QuotationChecker):
    def check_following(self, value):
        value = value.strip()
        return len(value) > 1 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "“" and value[-1] == "”")
            or (value[0] == "「" and value[-1] == "」")
        )


INSTRUCTION_DICT = {
    **instructions_registry.INSTRUCTION_DICT,
    "length_constraints:number_sentences": NumberOfSentences,
    "length_constraints:number_words": NumberOfWords,
    "startend:quotation": QuotationChecker,
}
