import re
from collections.abc import Mapping

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.models import (
    GradingOutput,
    PromptTemplate,
    PromptTemplateWithParseMap,
)


def _extract_xml_content(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else text


class CoherenceGradingOutput(GradingOutput):
    coherence_score: int


class CoherenceGrader:
    """
    Coherence grader taken from AidanBench: https://github.com/aidanmclaughlin/AidanBench/blob/main/benchmark/prompts.py
    """

    INSTRUCTION_KEY = "instruction"
    COMPLETION_KEY = "completion"
    PROMPT_TEMPLATES = {
        Language("en"): PromptTemplate(
            system_prompt="",  # noqa: E501
            user_prompt=f"""Your task is to evaluate the coherence and plausibility of an answer to a given question.

Question: <question>{{{INSTRUCTION_KEY}}}</question>
Answer: <answer>{{{COMPLETION_KEY}}}</answer>

Based on the following criteria, provide a Coherence and Plausibility Score on a scale of 0 - 100:

0-20: INCOHERENT/NONSENSICAL
- Answer is completely unrelated to the question
- Contains logical impossibilities or contradictions
- Makes claims that defy basic reality
- Shows no understanding of the question's context

21-40: SEVERELY FLAWED
- Major logical gaps or inconsistencies
- Significant misunderstanding of core concepts
- Contains partially relevant information but mostly incorrect
- May include some true statements but fails to form a coherent response

41-60: PARTIALLY COHERENT
- Shows basic understanding of the question
- Contains some valid points mixed with errors
- Logic is followable but may have weak connections
- Answer is relevant but may miss key aspects

61-80: MOSTLY COHERENT
- Demonstrates clear understanding of the question
- Logic is sound with minor gaps or inconsistencies
- Most claims are plausible and well-supported
- Forms a generally complete and relevant response

81-100: HIGHLY COHERENT
- Perfectly addresses the question
- Demonstrates complete logical consistency
- All claims are plausible and well-grounded
- Forms a comprehensive and precise response

IMPORTANT: Provide your final Coherence and Plausibility Score as a single integer between 0 and 100, enclosed in <coherence_score></coherence_score> XML tags. For example:
<coherence_score>75</coherence_score>

Do not include any additional text in your response.""",  # noqa: E501
        ),
    }

    def __init__(
        self,
        grading_model: BaseLLM,
        prompt_templates: Mapping[Language, PromptTemplateWithParseMap] = PROMPT_TEMPLATES,
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

    def grade(self, instruction: str, completion: str, language: Language) -> CoherenceGradingOutput:
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
        coherence_score = int(_extract_xml_content(raw_completion.completion, "coherence_score"))

        return CoherenceGradingOutput(
            coherence_score=coherence_score,
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
