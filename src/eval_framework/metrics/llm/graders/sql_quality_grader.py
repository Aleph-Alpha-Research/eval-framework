from collections.abc import Mapping
from typing import Any, Literal

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.metrics.llm_metrics.graders.models import (
    GradingOutput,
    PromptTemplateWithParseMap,
    parse_json_output,
)


class SqlQualityGradingOutput(GradingOutput):
    thought_process: str | None
    query_quality: Literal[1, 2, 3, 4, 5] | None


class SqlQualityGrader:
    PROMPT_TEMPLATE_KEYS = {
        "prompt": "prompt",
        "completion": "completion",
        "results": "results",
    }
    PROMPT_TEMPLATES = {
        Language("de"): PromptTemplateWithParseMap(
            system_prompt="""Deine Aufgabe ist es, die Arbeit eines Informatik-Studenten zu bewerten.
Der Student sollte eine SQL-Abfrage schreiben, die den angegebenen Anforderungen entspricht.

Benutze folgendes Schulnotensystem, um die Qualität der Arbeit des Studenten zu bewerten:
    'sehr gut': Außergewöhnlich effizient und genau, erfüllt die Ziele perfekt.
    'gut': Sehr effizient und genau, stimmt eng mit den Zielen überein.
    'befriedigend': Mäßig effizient und genau, stimmt mit geringfügigen Problemen mit den Zielen überein.
    'ausreichend': Ausreichend effizient und genau, erfüllt die Ziele mit einigen merklichen Schwächen.
    'mangelhaft': Begrenzte Effizienz oder Genauigkeit, erhebliche Probleme mit den Zielen.
    'ungenügend': Völlig daneben oder irrelevant.

Gebe deine Bewertung in folgendem JSON-Format:
{
    "thought_process": str (Bewerte die Qualität der geschriebenen SQL-Abfrage. Argumentiere in ein paar Sätzen.),
    "query_quality": Literal["sehr gut", "gut", "befriedigend", "ausreichend", "mangelhaft", "ungenügend"] (Eine Schulnote, die die Gesamtqualität der SQL-Abfrage darstellt. Eine sehr gute Antwort ist effizient und genau.)
}""",  # noqa: E501
            user_prompt=f"""
**Aufgabe**
{{{PROMPT_TEMPLATE_KEYS["prompt"]}}}

**Lösung des Studenten**
{{{PROMPT_TEMPLATE_KEYS["completion"]}}}

**Ergebnis der SQL-Abfrage des Studenten**
{{{PROMPT_TEMPLATE_KEYS["results"]}}}
""",
            parse_map={
                "sehr gut": 5,
                "gut": 4,
                "befriedigend": 3,
                "ausreichend": 2,
                "mangelhaft": 1,
                "ungenügend": 1,
            },
        ),
        Language("en"): PromptTemplateWithParseMap(
            system_prompt="""Your task is to evaluate the work of a computer science student.
The student should write a SQL query that meets the specified requirements.

Use the following grading system to evaluate the quality of the student's work:
A: Exceptionally efficient and accurate, perfectly meets objectives.
B: Highly efficient and accurate, closely aligns with objectives.
C: Moderately efficient and accurate, aligns with objectives with minor issues.
D: Limited efficiency or accuracy, significant issues with objectives.
F: Entirely off-target or irrelevant.

Provide your evaluation in the following JSON format:
{
    "thought_process": str (Evaluate the quality of the written SQL query. Argue in a few sentences.),
    "query_quality": Literal["A", "B", "C", "D", "F"] (A school grade that represents the overall quality of the SQL query. A very good answer is efficient and accurate.)
}
""",  # noqa: E501
            user_prompt=f"""
**Assignment**
{{{PROMPT_TEMPLATE_KEYS["prompt"]}}}

**The student's solution**
{{{PROMPT_TEMPLATE_KEYS["completion"]}}}

**Result of the SQL query from the student's solution**
{{{PROMPT_TEMPLATE_KEYS["results"]}}}
""",
            parse_map={
                "A": 5,
                "B": 4,
                "C": 3,
                "D": 2,
                "F": 1,
            },
        ),
    }

    def __init__(
        self,
        grading_model: StructuredOutputChatModel,
        prompt_templates: Mapping[Language, PromptTemplateWithParseMap] = PROMPT_TEMPLATES,
    ) -> None:
        self._grading_model = grading_model

        if not all(
            key in prompt_template.user_prompt
            for prompt_template in prompt_templates.values()
            for key in self.PROMPT_TEMPLATE_KEYS.keys()
        ):
            raise ValueError(
                f"At least one PromptTemplate is invalid, must contain '{list(self.PROMPT_TEMPLATE_KEYS.keys())}'."
            )
        self._prompt_templates = prompt_templates

    def grade(
        self,
        prompt: str,
        completion: str,
        result: list[Any] | None,
        language: Language,
    ) -> SqlQualityGradingOutput:
        prompt_template = language.language_config(self._prompt_templates)

        result_string = str(result) if result else "This query did not yield any results."
        messages = prompt_template.to_messages(
            [],
            [
                (self.PROMPT_TEMPLATE_KEYS["prompt"], prompt),
                (self.PROMPT_TEMPLATE_KEYS["completion"], completion),
                (
                    self.PROMPT_TEMPLATE_KEYS["results"],
                    result_string,
                ),
            ],
        )

        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return SqlQualityGradingOutput(
            thought_process=loaded_json.get("thought_process", None),
            query_quality=prompt_template.parse_map.get(str(loaded_json.get("query_quality", None)), None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
