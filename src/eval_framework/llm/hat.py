import functools
import logging
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import (
    ConcatCompression,
    Error,
    PromptTooLongException,
    RawCompletion,
    RawLoglikelihood,
)
from eval_framework.tasks.base import Sample
from eval_framework.tasks.utils import raise_errors
from eval_framework.utils.constants import RED, RESET
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message

logger = logging.getLogger(__name__)


def sample_argmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def fast_multinomial(p: torch.Tensor, /) -> torch.Tensor:
    """
    A slightly faster version of torch.multinomial(p, 1)
    See: https://github.com/pytorch/pytorch/issues/30968#issuecomment-859084590
    """
    cumulative_probs = p.cumsum(dim=-1)
    rand = torch.rand(p.shape[:-1], device=p.device).unsqueeze(-1)
    return (cumulative_probs >= rand).byte().argmax(dim=-1)


def sample_temperature(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0.0:
        return sample_argmax(logits)

    assert temperature > 0.0, "Temperature must be positive"
    scaled_logits = logits * (1 / temperature)
    token_probs = F.softmax(scaled_logits, dim=-1)
    return fast_multinomial(token_probs)


class HatLLM(BaseLLM):
    LLM_NAME: str
    DEFAULT_FORMATTER: Callable[[], BaseFormatter] | None = None
    SEQ_LENGTH: int | None = None

    def __init__(self, formatter: BaseFormatter | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.LLM_NAME, attn_implementation="flash_attention_2", trust_remote_code=True
        ).to(device=self.device, dtype=torch.bfloat16)
        logger.info(f"{RED}[ Model initialized --------------------- {RESET}{self.LLM_NAME} {RED}]{RESET}")

        if self.model.eos_token_id is not None:
            self.eos_text = self.model.splitter.decode([self.model.eos_token_id])
        self._formatter: BaseFormatter
        self._get_formatter(formatter)

    def _get_formatter(self, formatter: BaseFormatter | None = None) -> None:
        # if formatter is being set at initialization time, use it
        if formatter is not None:
            self._formatter = formatter
        # if formatter is not being set at initialization time, but DEFAULT_FORMATTER was specified, use it
        elif self.DEFAULT_FORMATTER is not None:
            self._formatter = self.DEFAULT_FORMATTER()
        # if formatter is not being set at initialization time and there is no default formatter and no chat formatter,
        # using ConcatFormatter
        else:
            raise ValueError("No formatter specified and no default formatter available.")

        logger.info(
            f"{RED}[ Using default formatter --------------------- {RESET}{self._formatter.__class__.__name__} {RED}]{RESET}"  # noqa: E501
        )

    def count_words(self, text: str, /) -> int:
        """Count the number of words in a string."""
        _, cumulative_word_lengths = self.model._prepare_input(text, add_llama_template=False)
        return len(cumulative_word_lengths) - 1

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        results = []
        for sample in samples:
            prompt = self._formatter.format(sample.messages, output_mode="string")
            prompt_input_ids, prompt_cumulative_word_lengths = self.model._prepare_input(
                prompt, add_llama_template=False
            )
            num_prompt_bytes = prompt_cumulative_word_lengths[-1]

            choices_log_probs: dict[str, float] = {}
            choices_log_probs_sequence_positions: dict[str, int] = {}
            error: Error | None = None

            for choice in sample.possible_completions or []:
                choice_input_ids, choice_cumulative_word_lengths = self.model._prepare_input(
                    choice, add_llama_template=False
                )
                num_choice_bytes = choice_input_ids.shape[-1]

                # Concatenate prompt and choice ids / cumulative lengths
                prompt_and_choice_input_ids = torch.cat((prompt_input_ids, choice_input_ids), dim=-1)
                prompt_and_choice_cumulative_word_lengths = torch.cat(
                    (
                        prompt_cumulative_word_lengths,
                        choice_cumulative_word_lengths[1:] + prompt_cumulative_word_lengths[-1],
                    ),
                    dim=-1,
                )

                total_bytes_count = prompt_and_choice_input_ids.shape[-1]
                max_bytes = min(filter(None, [self.SEQ_LENGTH, self.seq_length]))

                if max_bytes < total_bytes_count:
                    if raise_errors():
                        raise PromptTooLongException("Prompt exceeded context size.")
                    choices_log_probs = {}
                    choices_log_probs_sequence_positions = {}
                    error = Error(
                        error_class=PromptTooLongException.__name__,
                        message="Prompt and choice exceeded context size.",
                        traceback="",
                    )
                    break

                # Calculate log-likelihoods for each token in the completion
                sum_log_probs = self._model_log_probs(
                    input_ids=prompt_and_choice_input_ids,
                    cumulative_word_lengths=prompt_and_choice_cumulative_word_lengths,
                    num_choice_bytes=num_choice_bytes,
                )
                choices_log_probs.update({choice: sum_log_probs})
                choices_log_probs_sequence_positions.update({choice: num_choice_bytes})

            results.append(
                RawLoglikelihood(
                    prompt=prompt,
                    prompt_sequence_positions=num_prompt_bytes,
                    concat_compression=ConcatCompression.calculate(
                        sample.messages,
                        count_tokens=self.count_words,
                        choices=sample.possible_completions,
                    ),
                    loglikelihoods=choices_log_probs,
                    loglikelihoods_sequence_positions=choices_log_probs_sequence_positions,
                    raw_loglikelihood_error=error,
                )
            )
        return results

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float | None) -> torch.Tensor:
        last_logits = logits[:, -1, :]
        temperature = 1.0
        if temperature is None:
            return sample_argmax(last_logits)
        else:
            return sample_temperature(last_logits, temperature=temperature)

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        if temperature is None:
            logger.info("Using default temperature (0.), as no custom temperature value was provided")

        raw_completions = []
        for single_messages in messages:
            prompt = self._formatter.format(single_messages, output_mode="string")
            # add_special_tokens would add a second BOS token without explicitly setting it False
            input_ids, cumulative_word_lengths = self.model._prepare_input(prompt, add_llama_template=False)
            prompt_byte_count = input_ids.shape[-1]

            min_seq_length = min(filter(None, [self.seq_length, self.SEQ_LENGTH]))

            # Calculate the maximum number of tokens to generate
            max_bytes_to_generate = min_seq_length - prompt_byte_count
            # This is a hack based on typical compression ratios
            max_bytes = 5 * max_tokens
            max_bytes_to_generate = min(filter(None, [max_bytes_to_generate, max_bytes]))

            if max_bytes_to_generate < 1:
                if raise_errors():
                    raise PromptTooLongException("Prompt exceeded context size.")
                raw_completions.append(
                    RawCompletion(
                        prompt=prompt,
                        prompt_sequence_positions=prompt_byte_count,
                        completion="",
                        completion_sequence_positions=0,
                        raw_completion_error=Error(
                            error_class=PromptTooLongException.__name__,
                            message="Prompt exceeded context size.",
                            traceback="",
                        ),
                    )
                )
                continue

            completion, num_bytes = self._model_generate(
                input_ids,
                cumulative_word_lengths=cumulative_word_lengths,
                max_new_tokens=max_tokens,
                use_cache=True,
                stop_sequences=stop_sequences,
                sample_fn=functools.partial(self._sample, temperature=temperature),
            )

            raw_completions.append(
                RawCompletion(
                    prompt=prompt,
                    prompt_sequence_positions=prompt_byte_count,
                    concat_compression=ConcatCompression.calculate(
                        single_messages, count_tokens=self.count_words, completion=completion
                    ),
                    completion=completion,
                    completion_sequence_positions=num_bytes,
                )
            )
        return raw_completions

    def _model_generate(self, input_ids, cumulative_word_lengths, **kwargs):
        outputs = self.model.generate(input_ids, cumulative_seq_lengths_per_word=cumulative_word_lengths, **kwargs)
        completion = outputs.completion_text
        num_bytes = len(outputs.completion_logits)
        # These are not stripped out by the model generation code
        if self.model.eos_token_id is not None and completion.endswith(self.eos_text):
            completion = completion[: -len(self.eos_text)]
            num_bytes -= 1
        return completion, num_bytes

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        cumulative_seq_lengths_per_word: torch.Tensor,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
    ):
        if byte_position_ids is None:
            byte_position_ids = torch.arange(
                cumulative_seq_lengths_per_word[-1], device=input_ids.device, dtype=torch.int32
            ).unsqueeze(0)

        if word_position_ids is None:
            word_position_ids = torch.arange(
                len(cumulative_seq_lengths_per_word) - 1, device=input_ids.device, dtype=torch.int32
            ).unsqueeze(0)

        return self.model.forward(
            input_ids=input_ids,
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
            byte_position_ids=byte_position_ids,
            word_position_ids=word_position_ids,
        )

    @torch.no_grad()
    def _model_log_probs(
        self, input_ids: torch.Tensor, cumulative_word_lengths: torch.Tensor, num_choice_bytes: int
    ) -> float:
        outputs = self._model_forward(input_ids=input_ids, cumulative_seq_lengths_per_word=cumulative_word_lengths)

        target_logits = outputs.logits[:, -num_choice_bytes - 1 : -1, :].squeeze(0).float()
        target_ids = input_ids[:, -num_choice_bytes:].squeeze(0)
        assert len(target_logits) == num_choice_bytes

        target_distributions = torch.log_softmax(target_logits, dim=-1)
        target_logprobs = target_distributions[torch.arange(len(target_ids)), target_ids]
        return torch.sum(target_logprobs, dim=0).item()

    @property
    def seq_length(self) -> int | None:
        config = self.model.config
        return min(config.encoder_config.max_position_embeddings, config.decoder_config.max_position_embeddings)


class HatModel(HatLLM):
    LLM_NAME = "Aleph-Alpha/llama-tfree-hat-pretrained-7b-dpo"
    DEFAULT_FORMATTER = Llama3Formatter


class HatModelBase(HatLLM):
    LLM_NAME = "Aleph-Alpha/tfree-hat-pretrained-7b-base"
    DEFAULT_FORMATTER = ConcatFormatter
