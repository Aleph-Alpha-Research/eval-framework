from typing import Mapping

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.metrics.llm_metrics.graders.models import GradingOutput, PromptTemplate, parse_json_output


class LongContextGradingOutput(GradingOutput):
    answer_is_correct: bool | None


class LongContextGrader:
    COMPLETION_KEY = "completion"
    EXPECTED_OUTPUT_KEY = "expected_output"
    PROMPT_TEMPLATES = {
        Language("en"): PromptTemplate(
            system_prompt="""Your task is to classify if a text generation model's response matches the target response.

The response matches the target if the generation addresses the question correctly and provides the right information.

You must provide your evaluation in the following JSON format:
{
    "answer_is_correct": bool
}""",
            user_prompt=f"""
**Model Response**:
{{{COMPLETION_KEY}}}

---
**Expected Response**:
{{{EXPECTED_OUTPUT_KEY}}}""",
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

    def grade(
        self,
        expected_output: str,
        completion: str,
        language: Language,
    ) -> LongContextGradingOutput:
        try:
            prompt_template = language.language_config(self._prompt_templates)
        except Exception as _:
            prompt_template = Language("en").language_config(self._prompt_templates)

        messages = prompt_template.to_messages(
            [],
            [
                (self.COMPLETION_KEY, completion),
                (self.EXPECTED_OUTPUT_KEY, expected_output),
            ],
        )

        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return LongContextGradingOutput(
            answer_is_correct=loaded_json.get("answer_is_correct", None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
