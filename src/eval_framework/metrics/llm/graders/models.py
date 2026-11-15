import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from template_formatting.formatter import Message, Role

logger = logging.getLogger(__name__)


class PromptTemplate(BaseModel):
    system_prompt: str
    user_prompt: str

    @staticmethod
    def _format_string(template: str, format_dict: Mapping[str, str]) -> str:
        if format_dict:
            return template.format(**format_dict)
        return template

    def to_messages(
        self,
        system_key_value_pairs: list[tuple[str, str]],
        user_key_value_pairs: list[tuple[str, str]],
    ) -> Sequence[Message]:
        return [
            Message(
                role=Role.SYSTEM,
                content=self._format_string(
                    self.system_prompt,
                    {key: value for key, value in system_key_value_pairs},
                ),
            ),
            Message(
                role=Role.USER,
                content=self._format_string(
                    self.user_prompt,
                    {key: value for key, value in user_key_value_pairs},
                ),
            ),
        ]


class GradingOutput(BaseModel):
    judge_prompt: str
    judge_response: str


def parse_json_output(output: str) -> dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        parsed_json = match.group(0) if match else "{}"
        return json.loads(parsed_json)
    except (json.JSONDecodeError, ValueError) as e:
        logger.info(f"Warning: LLM judge produced an invalid JSON output, will treat it as an empty output. Error: {e}")
        return {}
