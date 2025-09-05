from collections.abc import Mapping
from enum import Enum

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.models import (
    GradingOutput,
    PromptTemplateWithParseMap,
    parse_json_output,
)


class MatchOutcome(str, Enum):
    A_WINS = "a_wins"
    DRAW = "draw"
    B_WINS = "b_wins"

    @property
    def payoff(self) -> tuple[float, float]:
        if self == self.A_WINS:
            return (1, 0)
        if self == self.DRAW:
            return (0.5, 0.5)
        return (0, 1)

    @staticmethod
    def from_rank_literal(rank: int) -> "MatchOutcome":
        match rank:
            case 1:
                return MatchOutcome.A_WINS
            case 2:
                return MatchOutcome.B_WINS
            case 3:
                return MatchOutcome.DRAW
            case _:
                raise ValueError(f"Got unexpected rank {rank}")


class ComparisonGradingOutput(GradingOutput):
    reasoning: str
    outcome: MatchOutcome


class ComparisonGrader:
    INSTRUCTION_KEY = "instruction"
    ANSWER_1_KEY = "answer_1"
    ANSWER_2_KEY = "answer_2"
    REASONING_KEY = "explanation"
    BETTER_ANSWER_KEY = "better_answer"
    PROMPT_TEMPLATES = {
        Language("de"): PromptTemplateWithParseMap(
            system_prompt=f"""Beachte die gegebene Aufgabe und dazugehörigen Antworten. Entscheide, welche Antwort besser ist, Antwort 1 oder Antwort 2. Gebe anschließend "Antwort 1 ist besser", "Antwort 2 ist besser" oder "Beide gleich" aus.

Eine gute Antwort ist:
1. Inhaltlich korrekt.
2. Beachtet die Anforderungen der Aufgabe präzise.
3. Ist im Rahmen der Aufgabenstellung kreativ und nicht repetetiv.
4. In der Sprache der Aufgabe verfasst.

Gebe die Antwort im folgenden json-Format:
{{
    "{REASONING_KEY}": str (Beschreibe in wenigen Sätzen (max. 5) die Unterschiede der beiden Antworten und begründe, warum eine der beiden Antworten besser ist oder warum die Antworten ähnlich gut sind.),
    "{BETTER_ANSWER_KEY}": Literal["Antwort 1 ist besser", "Antwort 2 ist besser", "Beide gleich"]
}}""",  # noqa: E501
            user_prompt=f"""Aufgabe:
{{{INSTRUCTION_KEY}}}
---
Antwort 1:
{{{ANSWER_1_KEY}}}
---
Antwort 2:
{{{ANSWER_2_KEY}}}""",
            parse_map={
                "Antwort 1 ist besser": MatchOutcome.A_WINS,
                "Antwort 2 ist besser": MatchOutcome.B_WINS,
                "Beide gleich": MatchOutcome.DRAW,
            },
        ),
        Language("en"): PromptTemplateWithParseMap(
            system_prompt=f"""Note the given task and the corresponding answers. Decide which answer is better, answer 1 or answer 2. Then output "Answer 1 is better", "Answer 2 is better" or "Both equal".

A good answer is:
1. correct in content.
2. follows the requirements of the task precisely.
3. is creative and not repetitive in the context of the task.
4. written in the same language as the task.

Enter the answer in the following json format:
{{
    "{REASONING_KEY}": str (Describe in a few sentences (max. 5) the differences between the two answers and give reasons why one of the two answers is better or why the answers are similarly good),
    "{BETTER_ANSWER_KEY}": Literal["Answer 1 is better", "Answer 2 is better", "Both equal"]
}}""",  # noqa: E501
            user_prompt=f"""Task:
{{{INSTRUCTION_KEY}}}
---
Answer 1:
{{{ANSWER_1_KEY}}}
---
Answer 2:
{{{ANSWER_2_KEY}}}""",
            parse_map={
                "Answer 1 is better": MatchOutcome.A_WINS,
                "Answer 2 is better": MatchOutcome.B_WINS,
                "Both equal": MatchOutcome.DRAW,
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
            self.INSTRUCTION_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()
        ) or not all(self.ANSWER_1_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()):
            raise ValueError(
                f"At least one PromptTemplate invalid, must contain '{self.ANSWER_1_KEY}' and '{self.INSTRUCTION_KEY}'."
            )
        self._prompt_templates = prompt_templates

    def grade(
        self, instruction: str, completion_1: str, completion_2: str, language: Language
    ) -> ComparisonGradingOutput:
        prompt_template = language.language_config(self._prompt_templates)

        messages = prompt_template.to_messages(
            [],
            [
                (self.INSTRUCTION_KEY, instruction),
                (self.ANSWER_1_KEY, completion_1),
                (self.ANSWER_2_KEY, completion_2),
            ],
        )

        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return ComparisonGradingOutput(
            reasoning=loaded_json.get(self.REASONING_KEY, None),
            outcome=prompt_template.parse_map.get(str(loaded_json.get(self.BETTER_ANSWER_KEY, None)), None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
