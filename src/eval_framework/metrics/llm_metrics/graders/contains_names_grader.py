from collections.abc import Mapping

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.metrics.llm_metrics.graders.models import GradingOutput, PromptTemplate, parse_json_output


class ContainsNamesGradingOutput(GradingOutput):
    contains_names: bool | None


class ContainsNamesGrader:
    COMPLETION_KEY = "completion"
    PROMPT_TEMPLATES = {
        Language("en"): PromptTemplate(
            system_prompt="""Your assignment is to assess whether text contains names of individuals.

Your assessment should be formatted in JSON as follows:
{
    "contains_names": bool (State with a true or false whether the text contains any names.)
}""",
            user_prompt=f"""{{{COMPLETION_KEY}}}""",
        ),
        Language("de"): PromptTemplate(
            system_prompt="""Deine Aufgabe ist es zu bewerten, ob ein Text Namen von Personen enthält.

Deine Bewertung sollte im JSON-Format wie folgt aussehen:
{
    "contains_names": bool (Gib an, ob der Text Namen enthält oder nicht.)
}""",
            user_prompt=f"""{{{COMPLETION_KEY}}}""",
        ),
        Language("fr"): PromptTemplate(
            system_prompt="""Votre tâche consiste à évaluer si un texte contient des noms de personnes.

Votre évaluation doit être formatée en JSON comme suit :
{
    "contains_names": bool (Indiquez si le texte contient des noms ou non.)
}""",
            user_prompt=f"""{{{COMPLETION_KEY}}}""",
        ),
    }

    def __init__(
        self,
        grading_model: StructuredOutputChatModel,
        prompt_templates: Mapping[Language, PromptTemplate] = PROMPT_TEMPLATES,
    ) -> None:
        self._grading_model = grading_model

        if not all(self.COMPLETION_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()):
            raise ValueError(f"At least one PromptTemplate is invalid, must contain '{self.COMPLETION_KEY}'.")
        self._prompt_templates = prompt_templates

    def grade(self, completion: str, language: Language) -> ContainsNamesGradingOutput:
        prompt_template = language.language_config(self._prompt_templates)
        messages = prompt_template.to_messages(
            [],
            [
                (self.COMPLETION_KEY, completion),
            ],
        )

        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return ContainsNamesGradingOutput(
            contains_names=loaded_json.get("contains_names", None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
