import copy
from collections.abc import Sequence
from typing import Literal, Protocol

import pytest

from template_formatting.formatter import ChatTemplate, Message, Role

package_exists_mc = pytest.importorskip("mistral_common")
package_exists_hf = pytest.importorskip("huggingface_hub")

if package_exists_mc and package_exists_hf:
    from huggingface_hub.utils import RepositoryNotFoundError

    from template_formatting.mistral_formatter import MagistralFormatter, MistralFormatter, MistralSerializer


class TypingStub(Protocol):
    template: ChatTemplate

    def format(  # type: ignore[override]
        self, messages: Sequence[Message], output_mode: Literal["list"] = "list"
    ) -> list[Message]: ...


class TestHFAssetRetrieval:
    def test_existing_repo(self) -> None:
        formatter = MagistralFormatter(llm_target="mistralai/Magistral-Small-2506")
        assert len(formatter.template.system_prompt) > 0

    def test_non_existing_repo(self) -> None:
        with pytest.raises(RepositoryNotFoundError):
            MagistralFormatter(llm_target="Qwen/phariachat")


class TestMistralFormatter:
    @pytest.fixture
    def mformatter(self) -> TypingStub:
        return MistralFormatter(llm_target="mistralai/Magistral-Small-2506")

    @pytest.fixture
    def chat(self) -> list[Message]:
        return [
            Message(
                content="You are a helpful assistant that provides clear and concise"
                " answers to general knowledge questions.",
                role=Role.SYSTEM,
            ),
            Message(content="What is the capital of France?", role=Role.USER),
            Message(content="The capital of France is Paris.", role=Role.ASSISTANT),
        ]

    def test_encoding_challenge(self, chat: list[Message]) -> None:
        """
        Current template-formatting repo introduces special tokens directly into
        the prompt (ex: <system_prompt>Be a kind agent</system_prompt>). The formatted
        prompt is later fed to the LLM for inference.

        Downside with this approach is that some tokenizers don't detect special tokens
        and map them to unique indices. Rather these are parsed along side the text. As
        the case with some of Mistral's tokenizers.

        This test serves as a validation check and further argumentation for the need of
        a customized MistralFormatter.
        """
        mistral_serializer = MistralSerializer(llm_target="mistralai/Magistral-Small-2506")

        mistral_msg_lst = mistral_serializer.convert_from_aa(msg_lst=chat)
        mistral_request_object = mistral_serializer.build_mistral_request(mistral_msg_lst=mistral_msg_lst)
        mistral_tokenized_object = mistral_serializer.tokenizer.instruct_tokenizer.encode_instruct(
            mistral_request_object
        )
        expected_token_ids = mistral_tokenized_object.tokens
        formatted_prompt_txt = mistral_tokenized_object.text
        formatted_token_ids = mistral_serializer.tokenizer.instruct_tokenizer.tokenizer.encode(
            formatted_prompt_txt, False, False
        )
        assert expected_token_ids != formatted_token_ids

    @staticmethod
    def __validate_request(request_msgs: list[Message], msg_lst: list[Message]) -> bool:
        request_check = [
            request_msgs[idx].role == msg_lst[idx].role and request_msgs[idx].content == msg_lst[idx].content
            for idx in range(0, len(request_msgs))
        ]
        return all(request_check)

    def test_base_request(self, mformatter: TypingStub, chat: list[Message]) -> None:
        output_openai_msgs: list[Message] = mformatter.format(messages=chat)
        assert self.__validate_request(request_msgs=output_openai_msgs, msg_lst=chat)

    def test_multiple_user_request(self, mformatter: TypingStub, chat: list[Message]) -> None:
        test_case = copy.copy(chat)
        test_case.insert(2, Message(content="What is the most beautiful monument in Paris ?", role=Role.USER))
        output_openai_msgs: list[Message] = mformatter.format(messages=test_case)
        test_case[1].content += f"\n\n{test_case[2].content}"
        test_case.pop(2)
        assert self.__validate_request(request_msgs=output_openai_msgs, msg_lst=test_case)

    def test_no_system_request(self, mformatter: TypingStub, chat: list[Message]) -> None:
        test_case = copy.copy(chat)
        test_case.pop(0)
        output_openai_msgs: list[Message] = mformatter.format(messages=test_case)
        assert self.__validate_request(request_msgs=output_openai_msgs, msg_lst=test_case)

    def test_complete_prompt(self, mformatter: TypingStub, chat: list[Message]) -> None:
        test_case = copy.copy(chat)
        test_case.pop()
        output_openai_msgs: list[Message] = mformatter.format(messages=test_case)
        assert self.__validate_request(request_msgs=output_openai_msgs, msg_lst=test_case)


class TestMagistralFormatter:
    @pytest.fixture
    def magistral_formatter(self) -> TypingStub:
        return MagistralFormatter(llm_target="mistralai/Magistral-Small-2506")

    def test_system_prompt_addition(self, magistral_formatter: TypingStub) -> None:
        chat = [Message(role=Role.USER, content="What is the capital of france ?")]
        message_lst = magistral_formatter.format(messages=chat)
        print(message_lst)
        assert message_lst[0].role == Role.SYSTEM
        assert message_lst[0].content == magistral_formatter.template.system_prompt

    def test_abandon_default_system_prompt(self, magistral_formatter: TypingStub) -> None:
        chat = [
            Message(
                role=Role.SYSTEM,
                content="This prompt is specific to math problems; complext prompts. You need to be smart"
                " to solve these problems.",
            ),
            Message(role=Role.USER, content="What is the gradient of a quadratic function"),
        ]
        message_lst = magistral_formatter.format(messages=chat)
        assert message_lst[0].role == Role.SYSTEM
        assert message_lst[0].content == chat[0].content
