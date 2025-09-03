from collections.abc import Mapping

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.models import GradingOutput, PromptTemplate, parse_json_output


class ConcisenessGradingOutput(GradingOutput):
    thought_process: str | None
    is_concise: bool | None


class ConcisenessGrader:
    INSTRUCTION_KEY = "instruction"
    COMPLETION_KEY = "completion"
    PROMPT_TEMPLATES = {
        Language("de"): PromptTemplate(
            system_prompt="""Deine Aufgabe ist es zu klassifizieren, ob eine von einem Textgenerator gelieferte Antwort kurz und prägnant ist.

Eine kurz und prägnante ("concise") Antwort ist eine Antwort, die knapp und auf den Punkt ist, ohne unnötige Details oder Ausführungen.

Gebe deine Bewertung in folgendem JSON-Format:
{
    "thought_process": str (Achte sehr genau auf die Antwort und argumentiere in ein paar Sätzen, ob die Antwort kurz und prägnant ("concise") ist oder nicht),
    "is_concise": bool
}""",  # noqa: E501
            user_prompt=f"""**Benutzeranweisung**
{{{INSTRUCTION_KEY}}}

---
**Antwort des Textgenerators**
{{{COMPLETION_KEY}}}""",
        ),
        Language("en"): PromptTemplate(
            system_prompt="""Your task is to classify if a text generation model's response is concise.

A concise response is one that is brief and to the point, without unnecessary details or elaboration.

You must provide your evaluation in the following JSON format:
{
    "thought_process": str (Pay very close attention to the response and argue whether the response is concise or not in a few sentences),
    "is_concise": bool
}""",  # noqa: E501
            user_prompt=f"""**User Instruction**:
{{{INSTRUCTION_KEY}}}

---
**Model Response**:
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
            self.INSTRUCTION_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()
        ) or not all(
            self.COMPLETION_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()
        ):
            raise ValueError(
                f"At least one PromptTemplate is invalid, must contain '{self.COMPLETION_KEY}' "
                "and '{self.INSTRUCTION_KEY}'."
            )
        self._prompt_templates = prompt_templates

    def grade(self, instruction: str, completion: str, language: Language) -> ConcisenessGradingOutput:
        try:
            prompt_template = language.language_config(self._prompt_templates)
        except Exception as _:
            prompt_template = Language("en").language_config(self._prompt_templates)

        messages = prompt_template.to_messages(
            [],
            [
                (self.INSTRUCTION_KEY, instruction),
                (self.COMPLETION_KEY, completion),
            ],
        )

        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return ConcisenessGradingOutput(
            thought_process=loaded_json.get("thought_process", None),
            is_concise=loaded_json.get("is_concise", None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
