from typing import Mapping

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.metrics.llm_metrics.graders.models import GradingOutput, PromptTemplate, parse_json_output


class SummarizationWorldKnowledgeGradingOutput(GradingOutput):
    contains_world_knowledge_thought_process: str | None
    contains_world_knowledge: bool | None


class SummarizationWorldKnowledgeGrader:
    REFERENCE_INPUT_KEY = "reference_input"
    COMPLETION_KEY = "completion"
    PROMPT_TEMPLATES = {
        Language("de"): PromptTemplate(
            system_prompt="""Deine Aufgabe ist es, zu bewerten ob eine Zusammenfassung Informationen, die über den Referenztext hinausgehen (auch genannt "Weltwissen") enthält.

Gebe die Antwort im folgenden JSON-Format:
{
    "contains_world_knowledge_thought_process": str (Achte sehr genau auf die Antwort und argumentiere in ein paar Sätzen, ob die Zusammenfassung Informationen enthält, die über den Referenztext hinausgehen),
    "contains_world_knowledge": bool (Enthält die Zusammenfassung Informationen die über den Referenztext hinausgehen?)
}""",  # noqa: E501
            user_prompt=f"""**Referenztext**
{{{REFERENCE_INPUT_KEY}}}

---
**Zusammenfassung**
{{{COMPLETION_KEY}}}""",
        ),
        Language("en"): PromptTemplate(
            system_prompt="""Your task is to evaluate a summary regarding whether it contains information that goes beyond the reference text (also known as "world knowledge").

You must provide your evaluation in the following JSON format:
{
    "contains_world_knowledge_thought_process": str (Pay very close attention to the summary and argue whether the response contains world knowledge or not in a few sentences),
    "contains_world_knowledge": bool (Does the summary contain information that goes beyond the reference text?),
}""",  # noqa: E501
            user_prompt=f"""**Reference Text**
{{{REFERENCE_INPUT_KEY}}}

---
**Summary**
{{{COMPLETION_KEY}}}""",
        ),
        Language("fr"): PromptTemplate(
            system_prompt="""Votre tâche consiste à évaluer une résumé pour déterminer s'il contient des informations qui vont au-delà du texte de référence (également appelé "connaissance du monde").

Vous devez fournir votre évaluation dans le format JSON suivant :
{
    "contains_world_knowledge_thought_process": str (Prêtez une attention particulière au résumé et argumentez si le résumé contient des informations qui vont au-delà du texte de référence ou non en quelques phrases),
    "contains_world_knowledge": bool (Le résumé contient-il des informations qui vont au-delà du texte de référence ?),
}""",  # noqa: E501
            user_prompt=f"""**Texte de référence**
{{{REFERENCE_INPUT_KEY}}}

---
**Résumé**
{{{COMPLETION_KEY}}}""",
        ),
    }

    def __init__(
        self,
        grading_model: StructuredOutputChatModel,
        prompt_templates: Mapping[Language, PromptTemplate] = PROMPT_TEMPLATES,
    ) -> None:
        self._grading_model = grading_model

        if not all(
            self.REFERENCE_INPUT_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()
        ) or not all(
            self.COMPLETION_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()
        ):
            raise ValueError(
                f"At least one PromptTemplate is invalid, must contain '{self.COMPLETION_KEY}' "
                "and '{self.REFERENCE_INPUT_KEY}'."
            )
        self._prompt_templates = prompt_templates

    def grade(
        self, reference_input: str, completion: str, language: Language
    ) -> SummarizationWorldKnowledgeGradingOutput:
        prompt_template = language.language_config(self._prompt_templates)

        messages = prompt_template.to_messages(
            [],
            [
                (self.REFERENCE_INPUT_KEY, reference_input),
                (self.COMPLETION_KEY, completion),
            ],
        )

        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return SummarizationWorldKnowledgeGradingOutput(
            contains_world_knowledge_thought_process=loaded_json.get("contains_world_knowledge_thought_process", None),
            contains_world_knowledge=loaded_json.get("contains_world_knowledge", None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
