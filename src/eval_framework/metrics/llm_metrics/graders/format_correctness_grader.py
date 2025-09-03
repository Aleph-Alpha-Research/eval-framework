from collections.abc import Mapping

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.metrics.llm_metrics.graders.models import (
    FOFOPromptTemplate,
    GradingOutput,
    parse_json_output,
)


class FormatCorrectnessOutput(GradingOutput):
    reasons: str | None
    format_correctness: int | None


class FormatCorrectnessGrader:
    INSTRUCTION_KEY = "<instruction>"
    COMPLETION_KEY = "<completion>"

    PROMPT_TEMPLATES = {
        Language("en"): FOFOPromptTemplate(
            system_prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
            user_prompt=f"""
            I would like you to create a leaderboard that evaluates the correctness of the format of answers from
            various large language models. To accomplish this, you will need to analyze the text prompts given to
            the models and their corresponding answers. Specifically, please ensure that your evaluation outputs are
            properly formatted as a json string. I will provide both the prompts and the responses for this purpose.\n

            Here is the prompt: {{
                "instruction": {INSTRUCTION_KEY}
            }}

            Here are the outputs of the models:
            [
                {{
                    "model": "model",
                    "answer": {COMPLETION_KEY}
                }},
            ]

            Please evaluate the formatting of the model’s responses by checking if they comply with the format
            specifications stated in the prompt. Perform a thorough format check and provide a detailed explanation
            for why the format is correct or incorrect. Your feedback should include the name of the model, followed
            by the format correctness status represented as ’1’ for correct and ’0’ for incorrect. Present your
            reasoning as bullet points within a single string for each model assessed. In other words, you should
            produce the following output:
            ```json
            [
                {{
                "model": <model-name>,
                "format_correctness": <correctness>,
                "reasons": <reasons-of-format-correctness>
                }}
            ]```

            Please note that your response should be a properly formatted JSON string and should not contain any
            additional content. We will load it directly as a JSON string in Python.
            """,
        ),
    }

    def __init__(
        self,
        grading_model: StructuredOutputChatModel,
        prompt_templates: Mapping[Language, FOFOPromptTemplate] = PROMPT_TEMPLATES,
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

    def grade(self, instruction: str, completion: str, language: Language) -> FormatCorrectnessOutput:
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
        reasons = loaded_json.get("reasons", None)
        if isinstance(reasons, list):
            reasons = ["• " + reason + "\n " for reason in reasons]
            reasons = "".join(reasons)
        elif isinstance(reasons, str):
            reasons = "• " + reasons + "\n "

        return FormatCorrectnessOutput(
            reasons=reasons,
            format_correctness=loaded_json.get("format_correctness", None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
