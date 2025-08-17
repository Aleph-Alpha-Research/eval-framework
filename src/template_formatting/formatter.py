# ruff: noqa: E501
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal, overload

from pydantic import BaseModel, Field, field_serializer, field_validator
from typing_extensions import override

try:
    from transformers import AutoTokenizer
except ImportError:
    print("template_formatting: `transformers` package is not installed, HFFormatter will not be available.")


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Property(Enum):
    ANSWER = "answer"
    THOUGHT = "thought"
    SOLUTION = "solution"


class Message(BaseModel):
    role: Role | None = Field(default=None)  # Optional due to compatibility with legacy finetuning format.
    property: Property | None = Field(default=None)
    content: str
    has_loss: bool | None = Field(default=None)
    type: str | None = Field(default=None)

    @field_serializer("role")
    def serialize_task_name(self, value: Role | None) -> str | None:
        if value is None:
            # Legacy finetuning format.
            return None
        return value.value

    @field_validator("role", mode="before")
    @classmethod
    def validate_task_name(cls, value: str | Role | None) -> Role | None:
        if value is None:
            # Legacy finetuning format.
            return None
        if isinstance(value, str):
            return Role(value)
        return value


@dataclass
class ChatTemplate:
    begin_of_text: str
    end_of_text: str
    begin_system_prompt: str
    system_prompt: str
    end_system_prompt: str
    begin_assistant_id: str
    end_assistant_id: str
    begin_user_id: str
    end_user_id: str


@dataclass
class ReasoningTemplate(ChatTemplate):
    begin_thought_id: str
    end_thought_id: str
    begin_solution_id: str
    end_solution_id: str
    begin_answer_id: str
    end_answer_id: str


class BaseFormatter:
    template: ChatTemplate | ReasoningTemplate
    strip_content: bool = False
    never_strip: bool = False

    def __init__(self) -> None:
        super().__init__()
        assert not (self.strip_content and self.never_strip), "strip_content and never_strip cannot be both True"

    @staticmethod
    def _verify_messages(messages: Sequence[Message]) -> None:
        grouped_messages = BaseFormatter._get_grouped_messages(messages)
        offset = int(grouped_messages[0][0].role == Role.SYSTEM)
        user_messages = grouped_messages[offset::2]
        assistant_messages = grouped_messages[offset + 1 :: 2]
        if grouped_messages[0][0].role is None:
            # Legacy finetuning format.
            assert all(m[0].role is None for m in user_messages)
        else:  # New format, assert role order.
            assert all(m[0].role == Role.USER for m in user_messages)
            assert all(m[0].role == Role.ASSISTANT for m in assistant_messages)

    @staticmethod
    def _verify_message_fields(messages: Sequence[Message], output_mode: str) -> None:
        if output_mode not in ("string", "list"):
            raise ValueError("Unsupported output_mode: choose 'string' or 'list'")

        for message in messages:
            if output_mode == "string":
                # eval-framework style
                if not hasattr(message, "role"):
                    raise ValueError("Message is missing 'role' property.")
                if (getattr(message, "type", None) is not None) or (getattr(message, "has_loss", None) is not None):
                    raise ValueError()

            elif output_mode == "list":
                # scaling style
                if not hasattr(message, "type") or not hasattr(message, "has_loss"):
                    raise ValueError("Message is missing 'type' or 'has_loss' property.")

    @staticmethod
    def _get_grouped_messages(messages: Sequence[Message]) -> Sequence[Sequence[Message]]:
        """
        Groups consecutive messages to meet two criteria, while preserving the
        order of each sequence item:
        - Role is identical in each group.
        - Each property occurs once in each group.
        """
        if not messages:
            return []

        grouped_messages = []
        current_group = [messages[0]]

        for message in messages[1:]:
            role = current_group[0].role
            group_props = set(i.property for i in current_group)
            if message.role == role and message.property not in group_props:
                current_group.append(message)
            else:
                grouped_messages.append(current_group)
                current_group = [message]

        grouped_messages.append(current_group)
        return grouped_messages

    @overload
    def format(self, messages: Sequence[Message], output_mode: Literal["string"]) -> str:
        pass

    @overload
    def format(self, messages: Sequence[Message], output_mode: Literal["list"]) -> list[Message]:
        pass

    def format(
        self, messages: Sequence[Message], output_mode: Literal["string", "list"] = "string"
    ) -> str | list[Message]:
        """
        Formats a list of messages using the provided template.
            output_mode: "string" returns a single concatenated string ('eval-framework' style),
                         "list" returns the messages with their content updated ('scaling' style).
        """
        self._verify_messages(messages)
        self._verify_message_fields(messages, output_mode)

        if output_mode not in {"string", "list"}:
            raise ValueError("Unsupported output_mode: choose 'string' or 'list'")

        if output_mode == "string":
            # Generate formatted strings for each message and join them.
            formatted_parts = (
                self._format_message(message, i == len(messages) - 1, output_mode) for i, message in enumerate(messages)
            )
            return self.template.begin_of_text + "".join(formatted_parts)
        else:
            # Create a new list of messages with updated content.
            new_messages: list[Message] = [message.model_copy(deep=True) for message in messages]
            for i, message in enumerate(new_messages):
                formatted_content = self._format_message(messages[i], i == len(messages) - 1, output_mode)
                message.content = formatted_content

            # Prepend the begin_of_text to the first message's content.
            if new_messages:
                new_messages[0].content = self.template.begin_of_text + new_messages[0].content
            return new_messages

    def _format_message(self, message: Message, is_last: bool, output_mode: Literal["string", "list"]) -> str:
        """
        Returns the formatted string for a single message.
        """
        if message.role == Role.SYSTEM:
            text = getattr(message, "content", "")
            if not text and hasattr(self.template, "system_prompt"):
                text = self.template.system_prompt
            if self.strip_content:
                text = text.strip()
            return f"{self.template.begin_system_prompt}{text}{self.template.end_system_prompt}"

        elif message.role == Role.USER:
            text = getattr(message, "content", "")
            if self.strip_content:
                text = text.strip()
            elif output_mode == "string":
                if is_last or self.template.end_user_id != "" and not self.never_strip:
                    text = text.strip()
            if output_mode == "string" or (output_mode == "list" and not is_last):
                # start assistant message after user message
                result = (
                    f"{self.template.begin_user_id}{text}{self.template.end_user_id}{self.template.begin_assistant_id}"
                )
            else:
                # default HF behavior for applying chat template with
                # `add_generation_prompt=False` and `continue_final_message=False` (as used in 'scaling')
                result = f"{self.template.begin_user_id}{text}{self.template.end_user_id}"
            return result

        elif message.role == Role.ASSISTANT:
            return self._format_assistant(message, is_last, output_mode)

        elif message.role is None:
            return getattr(message, "content", "")

        else:
            raise ValueError(f"Unsupported role: {message.role}")

    def _format_assistant(self, message: Message, is_last: bool, output_mode: Literal["string", "list"]) -> str:
        """
        Formats an assistant message based on its property.
        """
        text = getattr(message, "content", "")
        if self.strip_content:
            text = text.strip()

        if message.property is not None:
            raise ValueError("Message properties require ReasoningFormatter")

        else:
            result = text
            # In string mode (i.e., 'eval-framework'), omit end_assistant_id if this is the last message.
            # In list mode (i.e., 'scaling'), always append it.
            if output_mode == "list" or (output_mode == "string" and not is_last):
                result += self.template.end_assistant_id
            elif output_mode == "string":
                if not self.never_strip:
                    result = result.strip()
            else:
                raise ValueError(f"Unknown output_mode: {output_mode}")

        return result


class IdentityFormatter(BaseFormatter):
    template = ChatTemplate(
        begin_of_text="",
        end_of_text="",
        begin_system_prompt="",
        system_prompt="",
        end_system_prompt="",
        begin_assistant_id="",
        end_assistant_id="",
        begin_user_id="",
        end_user_id="",
    )


class ConcatFormatter(BaseFormatter):
    template = ChatTemplate(
        begin_of_text="",
        end_of_text="",
        begin_system_prompt="",
        system_prompt="",
        end_system_prompt="\n\n",
        begin_assistant_id="",
        end_assistant_id="\n\n",
        begin_user_id="",
        end_user_id="",
    )
    # new lines are handled on task level, so we don't need to strip content here


class Llama3Formatter(BaseFormatter):
    template = ChatTemplate(
        begin_of_text="<|begin_of_text|>",
        end_of_text="",
        begin_system_prompt="<|start_header_id|>system<|end_header_id|>\n\n",
        system_prompt="You are a helpful AI assistant",
        end_system_prompt="<|eot_id|>",
        begin_assistant_id="<|start_header_id|>assistant<|end_header_id|>\n\n",
        end_assistant_id="<|eot_id|>",
        begin_user_id="<|start_header_id|>user<|end_header_id|>\n\n",
        end_user_id="<|eot_id|>",
    )
    strip_content = True  # stripping content to ensure consistency with HF chat template formatter


class Qwen3Formatter(BaseFormatter):
    template = ChatTemplate(
        begin_of_text="",
        end_of_text="<|endoftext|>",
        begin_system_prompt="<|im_start|>system\n",
        system_prompt="",
        end_system_prompt="<|im_end|>\n",
        begin_assistant_id="<|im_start|>assistant\n",
        end_assistant_id="<|im_end|>\n",
        begin_user_id="<|im_start|>user\n",
        end_user_id="<|im_end|>\n",
    )
    never_strip = True

    def _format_message(self, message: Message, is_last: bool, output_mode: Literal["string", "list"]) -> str:
        result = super()._format_message(message, is_last, output_mode)
        if message.role == Role.USER and is_last and output_mode == "string":
            result = f"{result}<think>\n\n</think>\n\n"
        elif message.role == Role.ASSISTANT and is_last and output_mode == "string":
            result = (
                f"<think>\n\n</think>\n\n{result}{self.template.end_assistant_id}"
                f"{self.template.begin_assistant_id}<think>\n\n</think>\n\n"
            )
        return result


class HFFormatter(BaseFormatter):
    def __init__(self, hf_llm_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(hf_llm_name)
        assert self.tokenizer.chat_template is not None, "Chat template is not available for this HF model."

    @override
    def format(  # type: ignore[override]
        self, messages: Sequence[Message], output_mode: Literal["string", "list"] = "string"
    ) -> str:
        hf_chat = [{"role": m.role.value, "content": m.content} for m in messages if m.role is not None] or ValueError(
            "Message role cannot be None"
        )
        if output_mode == "string":
            # if the last message is an assistant message, don't add <|eot_id|> but continue the message
            # (i.e., cueing the assistant within the eval framework)
            does_prefilling = messages[-1].role == Role.ASSISTANT
            return self.tokenizer.apply_chat_template(
                hf_chat,
                tokenize=False,
                add_generation_prompt=not does_prefilling,
                continue_final_message=does_prefilling,
            )
        else:  # output_mode == "list": a little confusing here, because it always returns a string, but only used for tests, so ok
            return self.tokenizer.apply_chat_template(
                hf_chat, tokenize=False
            )  # will apply default values for add_generation_prompt and continue_final_message (both False)


class ReasoningFormatter(BaseFormatter):
    template: ReasoningTemplate
    remove_previous_thoughts: bool = False

    def __init__(self, base_formatter: type[BaseFormatter]) -> None:
        self.template = ReasoningTemplate(
            **asdict(base_formatter.template),
            begin_thought_id="<|begin_of_thought|>",
            end_thought_id="<|end_of_thought|>",
            begin_solution_id="<|begin_of_solution|>",
            end_solution_id="<|end_of_solution|>",
            begin_answer_id="<|begin_of_answer|>",
            end_answer_id="<|end_of_answer|>",
        )

    def _format_message(self, message: Message, is_last: bool, output_mode: Literal["string", "list"]) -> str:
        result = super()._format_message(message, is_last, output_mode)
        if message.role == Role.USER and output_mode == "string" and (is_last or not self.remove_previous_thoughts):
            result = f"{result}{self.template.begin_thought_id}"
        return result

    def _format_assistant(self, message: Message, is_last: bool, output_mode: Literal["string", "list"]) -> str:
        """
        Formats an assistant message based on its property.
        """
        text = getattr(message, "content", "")
        if self.strip_content:
            text = text.strip()

        if message.property == Property.THOUGHT:
            result = f"{text}{self.template.end_thought_id}{self.template.begin_solution_id}"

        elif message.property == Property.SOLUTION:
            result = f"{text}{self.template.begin_answer_id}"

        elif message.property == Property.ANSWER:
            result = (
                f"{text}{self.template.end_answer_id}{self.template.end_solution_id}{self.template.end_assistant_id}"
            )
            if is_last:
                result = f"{result}{self.template.end_of_text}"

        elif message.property is None:
            result = text
            # In string mode (i.e., 'eval-framework'), omit end_assistant_id if this is the last message.
            # In list mode (i.e., 'scaling'), always append it.
            if output_mode == "list" or (output_mode == "string" and not is_last):
                result += self.template.end_assistant_id
            elif output_mode == "string":
                if not self.never_strip:
                    result = result.strip()
            else:
                raise ValueError(f"Unknown output_mode: {output_mode}")

        else:
            raise ValueError(f"Unsupported property: {message.property}")

        return result

    @staticmethod
    def _verify_messages(messages: Sequence[Message]) -> None:
        # Verify role order.
        BaseFormatter._verify_messages(messages)
        # Verify assistant message sequence.
        for group in BaseFormatter._get_grouped_messages(messages):
            if group[0].role == Role.ASSISTANT:
                if group[0].property is None:
                    for msg in group:
                        assert msg.property is None, "Assistant message group contains unexpected property combination."
                    continue
                if len(group) == 1:
                    assert group[0].property == Property.THOUGHT
                elif len(group) == 2:
                    assert group[0].property == Property.THOUGHT
                    assert group[1].property == Property.SOLUTION
                elif len(group) == 3:
                    assert group[0].property == Property.THOUGHT
                    assert group[1].property == Property.SOLUTION
                    assert group[2].property == Property.ANSWER
                else:
                    raise ValueError("Assistant message group is too long")

    def _validate_output(self, output_str: str) -> tuple[str, ValueError | None]:
        """Validate the output string according to following cases:
        A) Duplicate Tokens,
        B) Missing Tokens,
        C) Wrong Order,
        D) Still Thinking,
        E) Incomplete,
        F) valid.
        """
        required_tokens = [
            self.template.end_thought_id,
            self.template.begin_solution_id,
            self.template.end_solution_id,
            self.template.begin_answer_id,
            self.template.end_answer_id,
        ]

        # --- Case A: Duplicate tokens ---
        for token in [self.template.begin_thought_id] + required_tokens:
            count = output_str.count(token)
            if count > 1:
                return "error", ValueError(f"Duplicate tokens detected: '{token}' appears {count} times.")

        # --- Case B: Wrong Order ---
        last_index = -1
        missing_tokens = []
        for token in required_tokens:
            index = output_str.find(token)
            if index == -1:  # Token is missing
                missing_tokens.append(token)
            else:
                if missing_tokens:  # Other token found before missing token
                    first = missing_tokens[0]
                    return "error", ValueError(f"Missing token: Expected '{first}' but found '{token}' instead.")
                if index < last_index:  # Token is out of order
                    return "error", ValueError(f"Incorrect token order: '{token}' appears before expected.")
                last_index = index

        # --- Case C: No end_thought_id ---
        if self.template.end_thought_id in missing_tokens:
            return "not_finished_thinking", None  # Incomplete thinking (Case C)

        # --- Case D: Correct Order but incomplete ---
        elif missing_tokens:
            return "incomplete", None  # Incomplete output (Case D)

        # --- Case E: Valid ---
        else:
            return "valid", None  # valid (Case E)

    def _parse_output(self, output_str: str, thought_only: bool = False) -> dict[str, str]:
        """
        Extracts reasoning, solution, and final answer texts.
        - If 'thought_only=True', extracts only the reasoning part.
        - Uses regex to handle partial/incomplete outputs.
        """

        if thought_only:
            # Allow incomplete outputs (end_of_text is optional)
            pattern = (
                re.escape(self.template.begin_thought_id)
                + r"(.*?)"
                + re.escape(self.template.end_thought_id)
                + r".*?"
                + re.escape(self.template.end_of_text)
                + r"$"  # <-- Allows anything before <|end_of_text|>
            )
        else:
            # Full extraction pattern
            pattern = (
                re.escape(self.template.begin_thought_id)
                + r"(.*?)"
                + re.escape(self.template.end_thought_id)
                + re.escape(self.template.begin_solution_id)
                + r"(.*?)"
                + re.escape(self.template.end_solution_id)
                + re.escape(self.template.begin_answer_id)
                + r"(.*?)"
                + re.escape(self.template.end_answer_id)
                + r"(?:\s*"
                + re.escape(self.template.end_of_text)
                + r")?"
                + r"$"
            )

        # Use re.search for partial extraction
        match = re.search(pattern, output_str, re.DOTALL)
        if not match:
            raise ValueError("Parsing failed: Output format does not match expected structure.")

        # Safely extract each part (handles missing sections)
        reasoning_text = match.group(1).strip() if match.group(1) else ""
        solution_text = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""
        final_answer_text = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else ""

        # Return structured Messages
        return {"thought": reasoning_text, "solution": solution_text, "answer": final_answer_text}

    def parse(self, output_str: str) -> tuple[dict[str, str], ValueError | None]:
        (status, error) = self._validate_output(output_str)
        match status:
            case "error":
                return {}, error
            case "not_finished_thinking":
                output_str_without_end = output_str.replace(self.template.end_of_text, "")
                output_str_extended = output_str_without_end + self.template.end_thought_id + self.template.end_of_text
                return self._parse_output(output_str_extended, thought_only=True), None
            case "incomplete":
                return self._parse_output(output_str, thought_only=True), None
            case "valid":
                return self._parse_output(output_str), None
            case _:
                raise ValueError("Invalid status")


class Qwen3ReasoningFormatter(ReasoningFormatter):
    template: ReasoningTemplate
    never_strip = True
    remove_previous_thoughts = True

    def __init__(self, base_formatter: type[BaseFormatter] = Qwen3Formatter, disable_thinking: bool = False) -> None:
        if not issubclass(base_formatter, Qwen3Formatter):
            raise ValueError("Qwen3ReasoningFormatter requires Qwen3Formatter as base_formatter")

        self.base_formatter = base_formatter()
        self.disable_thinking = disable_thinking
        self.template = ReasoningTemplate(
            **asdict(base_formatter.template),
            begin_thought_id="<think>\n",
            end_thought_id="\n</think>\n\n",
            begin_solution_id="",
            end_solution_id="",
            begin_answer_id="",
            end_answer_id="",
        )

    def _format_message(self, message: Message, is_last: bool, output_mode: Literal["string", "list"]) -> str:
        if self.disable_thinking:
            return self.base_formatter._format_message(message, is_last, output_mode)
        else:
            result = super()._format_message(message, is_last, output_mode)
            if message.role == Role.ASSISTANT and is_last and output_mode == "string":
                result = (
                    f"{self.template.begin_thought_id}{self.template.end_thought_id}{result}"
                    f"{self.template.end_assistant_id}{self.template.begin_assistant_id}{self.template.begin_thought_id}"
                )
            return result

    def _format_assistant(self, message: Message, is_last: bool, output_mode: Literal["string", "list"]) -> str:
        if self.disable_thinking:
            return self.base_formatter._format_assistant(message, is_last, output_mode)
        else:
            return super()._format_assistant(message, is_last, output_mode)


def get_formatter(llm_name: str) -> BaseFormatter:
    llm_name = llm_name.lower()
    if "ng_7b" in llm_name or "pharia" in llm_name:
        print("Use LuminousNextgenFormatter")
        return Llama3Formatter()
    elif "llama-3" in llm_name:
        print("Use Llama3Formatter")
        return Llama3Formatter()
    else:
        print("Use ConcatFormatter")
        return ConcatFormatter()
