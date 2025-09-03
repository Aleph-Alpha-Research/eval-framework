import re

import nltk

from eval_framework.metrics.base import (
    BaseMetric,
    MetricResult,
)
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric

ALPHABETS = "([A-Za-z])"
PREFIXES = "(Mr|St|Mrs|Ms|Dr|www)[.]"
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov|edu|me)"
DIGITS = "([0-9])"
MULTIPLE_DOTS = r"\.{2,}"


class WordCounterMetricContext(BaseMetricContext):
    comparison: str
    word_count: int


class WordCounter(BaseMetric[Completion]):
    NAME = "Word Count"

    @staticmethod
    def _count_words(text: str) -> int:
        tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text)
        num_words = len(tokens)
        return num_words

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, WordCounterMetricContext)

        assert context.comparison in ["less than", "at least"], f"'comparison' is not valid: {context.comparison}"

        num_words = self._count_words(response.completion)
        if context.comparison == "less than":
            valid_word_count = num_words < context.word_count
        if context.comparison == "at least":
            valid_word_count = num_words >= context.word_count

        return [
            MetricResult(
                metric_name=self.NAME, value=float(valid_word_count), higher_is_better=True, error=response.error
            )
        ]


class SentenceCounterMetricContext(BaseMetricContext):
    comparison: str
    sentence_count: int


class SentenceCounter(BaseMetric[Completion]):
    NAME = "Sentence Count"

    @staticmethod
    def _count_sentences(text: str) -> int:
        # Note that nltk.tokenize.sent_tokenize would be a straightforward alternative but is also not ideal. Example:
        #
        # "Mr. Jones gave me $10,000.00... And then he left. Numbers 5...10. Numbers 5..10. Review: bad food,
        #    bad service,..., so I'd miss it."
        #
        # this: ['Mr. Jones gave me $10,000.00...', 'And then he left.', 'Numbers 5...', '10.', 'Numbers 5..', '10.',
        #    'Review: bad food, bad service,...', ", so I'd miss it."].
        # nltk: ['Mr. Jones gave me $10,000.00... And then he left.', 'Numbers 5...10.',
        #    "Numbers 5..10. Review: bad food, bad service,..., so I'd miss it."]

        text = f" {text} "
        text = text.replace("\n", " ")
        text = re.sub(PREFIXES, "\\1<prd>", text)
        text = re.sub(WEBSITES, "<prd>\\1", text)
        text = re.sub(DIGITS + "[.]" + DIGITS, "\\1<prd>\\2", text)
        text = re.sub(
            MULTIPLE_DOTS,
            lambda match: "<prd>" * len(match.group(0)) + "<stop>",
            text,
        )
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub(r"\s" + ALPHABETS + "[.] ", " \\1<prd> ", text)
        text = re.sub(ACRONYMS + " " + STARTERS, "\\1<stop> \\2", text)
        text = re.sub(
            ALPHABETS + "[.]" + ALPHABETS + "[.]" + ALPHABETS + "[.]",
            "\\1<prd>\\2<prd>\\3<prd>",
            text,
        )
        text = re.sub(ALPHABETS + "[.]" + ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + SUFFIXES + "[.] " + STARTERS, " \\1<stop> \\2", text)
        text = re.sub(" " + SUFFIXES + "[.]", " \\1<prd>", text)
        text = re.sub(" " + ALPHABETS + "[.]", " \\1<prd>", text)
        text = text.replace(".”", "”.")
        text = text.replace('."', '".')
        text = text.replace('!"', '"!')
        text = text.replace('?"', '"?')
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = [s.strip() for s in sentences]
        if sentences and not sentences[-1]:
            sentences = sentences[:-1]
        return len(sentences)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, SentenceCounterMetricContext)

        assert context.comparison in ["less than", "at least"], f"'comparison' is not valid: {context.comparison}"

        num_sentences = self._count_sentences(response.completion)
        if context.comparison == "less than":
            valid_sentence_count = num_sentences < context.sentence_count
        elif context.comparison == "at least":
            valid_sentence_count = num_sentences >= context.sentence_count

        return [
            MetricResult(
                metric_name=self.NAME, value=float(valid_sentence_count), higher_is_better=True, error=response.error
            )
        ]


class ParagraphCounterMetricContext(BaseMetricContext):
    comparison: str
    paragraph_count: int


class ParagraphCounter(BaseMetric[Completion]):
    NAME = "Paragraph Count"

    @staticmethod
    def _count_paragraphs(text: str) -> int:
        paragraphs = re.split(r"\s?\n\n\s?", text)
        return len(paragraphs)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, ParagraphCounterMetricContext)

        assert context.comparison in ["less than", "at least"], f"'comparison' is not valid: {context.comparison}"

        num_paragraphs = self._count_paragraphs(response.completion)
        if context.comparison == "less than":
            valid_paragraph_count = num_paragraphs < context.paragraph_count
        elif context.comparison == "at least":
            valid_paragraph_count = num_paragraphs >= context.paragraph_count

        return [
            MetricResult(
                metric_name=self.NAME, value=float(valid_paragraph_count), higher_is_better=True, error=response.error
            )
        ]


class ResponseToOriginalLengthRatio(BaseMetric[Completion]):
    NAME = "Response to Original Length Ratio"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        len_original = len(response.last_user_instruction)
        if len_original > 0:
            score = len(response.completion) / len_original
            return [MetricResult(metric_name=self.NAME, value=score, higher_is_better=False, error=response.error)]
        else:
            return []
