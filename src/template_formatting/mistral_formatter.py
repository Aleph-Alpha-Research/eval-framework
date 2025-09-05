from collections.abc import Sequence
from typing import Literal, cast

from huggingface_hub import hf_hub_download, try_to_load_from_cache

# mistral's api specific imports
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest
from mistral_common.tokens.tokenizers.base import InstructTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# package level imports
from .formatter import BaseFormatter, ChatTemplate, Message, Role


class MistralSerializer:
    def __init__(self, llm_target: str):
        self.tokenizer = MistralTokenizer.from_hf_hub(llm_target)

    def get_tokenizer(self) -> InstructTokenizer:
        return self.tokenizer.instruct_tokenizer

    @staticmethod
    def convert_to_aa(msg_lst: Sequence[SystemMessage | UserMessage | AssistantMessage]) -> Sequence[Message]:
        translated_messages: list[Message] = []
        for msg in msg_lst:
            match msg.role:
                case "system":
                    translated_messages.append(Message(role=Role.SYSTEM, content=msg.content))
                case "user":
                    translated_messages.append(Message(role=Role.USER, content=msg.content))
                case "assistant":
                    translated_messages.append(Message(role=Role.ASSISTANT, content=msg.content))
                case _:
                    raise ValueError("Role not supported")
        return translated_messages

    @staticmethod
    def convert_from_aa(msg_lst: Sequence[Message]) -> Sequence[SystemMessage | UserMessage | AssistantMessage]:
        translated_messages: list[SystemMessage | UserMessage | AssistantMessage] = []
        for idx, msg in enumerate(msg_lst):
            match msg.role:
                case Role.SYSTEM:
                    translated_messages.append(SystemMessage(content=msg.content))
                case Role.USER:
                    translated_messages.append(UserMessage(content=msg.content))
                case Role.ASSISTANT:
                    is_completion_request = idx == (len(msg_lst) - 1)  # insturcts model to complete
                    translated_messages.append(AssistantMessage(content=msg.content, prefix=is_completion_request))
                case _:
                    raise ValueError("Role not supported")
        return translated_messages

    def build_mistral_request(
        self, mistral_msg_lst: Sequence[SystemMessage | UserMessage | AssistantMessage]
    ) -> InstructRequest:
        # build chat request
        request: ChatCompletionRequest = ChatCompletionRequest(messages=mistral_msg_lst)
        # validate pydantic fields
        self.tokenizer._chat_completion_request_validator.validate_request(request)
        # merge same class messages
        instruct_request = self.tokenizer._instruct_request_normalizer.from_chat_completion_request(request)
        return instruct_request


class MistralFormatter(BaseFormatter):
    def __init__(self, llm_target: str) -> None:
        self.bridge_operator = MistralSerializer(llm_target=llm_target)

    def format(  # type: ignore[override]
        self, messages: Sequence[Message], output_mode: Literal["list"] = "list"
    ) -> list[Message]:
        """
        MistralFormatter intentionally restricts output_mode to 'list' only.

        This restriction exists because Mistral's tokenization requires special handling
        that bypasses traditional string-based formatting to preserve token boundaries.
        String mode would break the careful tokenization that Mistral's API provides.

        The type: ignore[override] is intentional; we're deliberately narrowing the
        interface.

        Args:
            messages: Sequence of messages to format
            output_mode: Must be "list" - string mode is not supported

        Returns:
            List of validated messages with plain text content

        Raises:
            ValueError: If output_mode is not "list"
        """
        # run back and forth translation and validate messages using mistral's API
        if output_mode not in {"list"}:
            raise ValueError("Unsupported output_mode: choose 'list'")

        mistral_msg_lst = self.bridge_operator.convert_from_aa(msg_lst=messages)
        mistral_request_object = self.bridge_operator.build_mistral_request(mistral_msg_lst=mistral_msg_lst)
        aa_msg_lst = self.bridge_operator.convert_to_aa(msg_lst=mistral_request_object.messages)

        # run validation using AA API
        self._verify_messages(aa_msg_lst)
        self._verify_message_fields(aa_msg_lst, "list")

        return cast(list, aa_msg_lst)


class MagistralFormatter(MistralFormatter):
    # these fields are not defined; left to MistralAPI to define; we only leverage system-prompt field
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

    def __init__(self, llm_target: str, sys_prompt_fname: str = "SYSTEM_PROMPT.txt") -> None:
        """
        sys_prompt_fname: name of folder on Magistral model card
        """

        def read_file(fname: str) -> str:
            with open(fname) as f:
                return f.read().strip()

        super().__init__(llm_target)
        prompt_path = try_to_load_from_cache(repo_id=llm_target, filename=sys_prompt_fname)
        if isinstance(prompt_path, str):
            self.template.system_prompt = read_file(fname=prompt_path)
        else:
            try:
                prompt_path = hf_hub_download(repo_id=llm_target, filename=sys_prompt_fname)
                self.template.system_prompt = read_file(fname=prompt_path)
            except Exception as e:
                raise e

    def format(  # type: ignore[override]
        self, messages: Sequence[Message], output_mode: Literal["list"] = "list"
    ) -> list[Message]:
        """
        MagistralFormatter extends MistralFormatter with automatic system prompt injection.

        Inherits the same 'list'-only restriction from MistralFormatter for the same
        tokenization reasons.
        """
        if output_mode not in {"list"}:
            raise ValueError("Unsupported output_mode: choose 'list'")

        if messages[0].role != Role.SYSTEM:
            input_messages = [Message(role=Role.SYSTEM, content=self.template.system_prompt), *messages]
        else:
            input_messages = cast(list, messages)

        return super().format(messages=input_messages)
