import logging
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from eval_framework.constants import RED, RESET
from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import Error, PromptTooLongException, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from eval_framework.tasks.utils import raise_errors, redis_cache
from template_formatting.formatter import BaseFormatter, ConcatFormatter, HFFormatter, Llama3Formatter, Message

logger = logging.getLogger(__name__)


class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Tokenizer, stop_sequences: list[str], prompt_token_count: int) -> None:
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.prompt_token_count = prompt_token_count
        # (relatively weak) upper bound for the number of tokens that
        # need to be decoded to check for stop sequences
        self.token_history_length = max(map(len, stop_sequences))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        sequence = input_ids[0].tolist()
        sequence = sequence[self.prompt_token_count :]
        if len(sequence) > self.token_history_length:
            sequence = sequence[-self.token_history_length :]
        decoded_text = self.tokenizer.decode(sequence, skip_special_tokens=True)

        for stop_sequence in self.stop_sequences:
            if stop_sequence in decoded_text:
                return True
        return False


class RepeatedTokenSequenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Tokenizer, completion_start_index: int) -> None:
        self.tokenizer = tokenizer
        # Initialize with an empty string to store the last line
        self.last_line = ""
        self.completion_start_index = completion_start_index
        # self.newline_token_id = tokenizer.encode('\n')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.Tensor:
        # Convert token ids to tokens
        tokens = self.tokenizer.decode(input_ids[0][self.completion_start_index :])

        # Join tokens to form the current text
        current_text = "".join(tokens)
        # Split text into lines
        lines = current_text.split("\n")

        # Check if the last full line (ignoring the last if it's incomplete) is repeated
        if len(lines) > 1 and lines[-2] == lines[-1] and not (lines[-1] == "" and lines[-2] == ""):
            return torch.BoolTensor([True]).to(input_ids.device)  # Stop generation if repeated line is found

        return torch.BoolTensor([False]).to(input_ids.device)


class HFLLM(BaseLLM):
    LLM_NAME: str
    DEFAULT_FORMATTER: Callable[[], BaseFormatter] | None = None
    SEQ_LENGTH: int | None = None

    def __init__(self, formatter: BaseFormatter | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.LLM_NAME, device_map="auto")
        logger.info(f"{RED}[ Model initialized --------------------- {RESET}{self.LLM_NAME} {RED}]{RESET}")
        self._set_formatter(formatter)

    def _set_formatter(self, formatter: BaseFormatter | None = None) -> None:
        # if formatter is being set at initialization time, use it
        if formatter is not None:
            self._formatter = formatter
        # if formatter is not being set at initialization time, but DEFAULT_FORMATTER was specified, use it
        elif self.DEFAULT_FORMATTER is not None:
            self._formatter = self.DEFAULT_FORMATTER()
        # if formatter is not being set at initialization time and there is no default formatter,
        # using HF chat formatter if exists
        elif self.tokenizer.chat_template is not None:
            self._formatter = HFFormatter(self.LLM_NAME)
        # if formatter is not being set at initialization time and there is no default formatter and no chat formatter,
        # using ConcatFormatter
        else:
            raise ValueError("No formatter specified and no default formatter available.")

        logger.info(
            f"{RED}[ Using default formatter --------------------- {RESET}{self._formatter.__class__.__name__} {RED}]{RESET}"  # noqa: E501
        )

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        if temperature is None:
            effective_temperature = 0.0  # Current default, TODO: refactor to use model's default
            logger.info(
                f"Using default temperature value: {effective_temperature} as no custom temperature value was provided"
            )
        else:
            effective_temperature = temperature

        raw_completions = []
        for single_messages in messages:
            # format
            prompt = self._formatter.format(single_messages, output_mode="string")
            # add_special_tokens would add a second BOS token without explicitly setting it False
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

            prompt_token_count = len(inputs["input_ids"][0])
            pad_token_id = self.tokenizer.eos_token_id

            # Prepare stopping criteria
            stopping_criteria = StoppingCriteriaList()
            if stop_sequences is not None:
                stopping_criteria.append(StopSequenceCriteria(self.tokenizer, stop_sequences, prompt_token_count))  # type: ignore[attr-defined]

            stopping_criteria.append(  # type: ignore[attr-defined]
                RepeatedTokenSequenceCriteria(
                    self.tokenizer,
                    prompt_token_count,
                )
            )

            min_seq_length = min(filter(None, [self.seq_length, self.SEQ_LENGTH]))

            # Calculate the maximum number of tokens to generate
            max_tokens_to_generate = min_seq_length - prompt_token_count
            # If max_tokens is specified, use the smaller of the two
            max_tokens_to_generate = min(filter(None, [max_tokens_to_generate, max_tokens]))

            if max_tokens_to_generate < 1:
                if raise_errors():
                    raise PromptTooLongException("Prompt exceeded context size.")
                raw_completions.append(
                    RawCompletion(
                        prompt=prompt,
                        prompt_sequence_positions=prompt_token_count,
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

            completion, completion_token_count = self._model_generate(
                redis_key=(prompt, stop_sequences, max_tokens_to_generate, effective_temperature),
                prompt_token_count=prompt_token_count,
                inputs=inputs["input_ids"],
                max_new_tokens=max_tokens_to_generate,
                stopping_criteria=stopping_criteria,
                num_return_sequences=1,
                pad_token_id=pad_token_id,
                return_dict_in_generate=False,
                output_scores=False,
                do_sample=effective_temperature > 0,
                temperature=effective_temperature if effective_temperature > 0 else None,
            )

            raw_completions.append(
                RawCompletion(
                    prompt=prompt,
                    prompt_sequence_positions=prompt_token_count,
                    completion=completion,
                    completion_sequence_positions=completion_token_count,
                )
            )
        return raw_completions

    @redis_cache(
        version_id="v8",
        args_key_func=lambda _, kwargs: str(kwargs["redis_key"]),
        load_func=lambda d: tuple(d),
    )
    def _model_generate(self, redis_key: Any, prompt_token_count: int, **kwargs: Any) -> tuple[str, int]:
        outputs = self.model.generate(**kwargs)[0]
        completion = self.tokenizer.decode(outputs[prompt_token_count:], skip_special_tokens=True)

        if kwargs["stopping_criteria"][0].__class__.__name__ == "StopSequenceCriteria":
            for stop_sequence in kwargs["stopping_criteria"][0].stop_sequences:
                completion = completion.split(stop_sequence)[0]
        return completion, len(outputs[prompt_token_count:])

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        results = []
        for sample in samples:
            # format
            prompt = self._formatter.format(sample.messages, output_mode="string")
            choices_log_probs: dict[str, float] = {}
            choices_log_probs_sequence_positions: dict[str, float] = {}
            error: Error | None = None

            for choice in sample.possible_completions or []:
                num_choice_tokens = len(self.tokenizer.encode(choice, add_special_tokens=False))
                prompt_and_choice = f"{prompt}{choice}"

                total_tokens_count = len(self.tokenizer.encode(prompt_and_choice, add_special_tokens=False))

                min_max_tokens = min(filter(None, [self.SEQ_LENGTH, self.seq_length]))

                if min_max_tokens < total_tokens_count:
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
                else:
                    # Calculate log-likelihoods for each token in the completion
                    sum_log_probs = self._model_log_probs(prompt_and_choice, num_choice_tokens)

                choices_log_probs.update({choice: sum_log_probs})
                choices_log_probs_sequence_positions.update({choice: num_choice_tokens})

            results.append(
                RawLoglikelihood(
                    prompt=prompt,
                    prompt_sequence_positions=len(self.tokenizer.encode(prompt, add_special_tokens=False)),
                    loglikelihoods=choices_log_probs,
                    loglikelihoods_sequence_positions=choices_log_probs_sequence_positions,
                    raw_loglikelihood_error=error,
                )
            )
        return results

    @redis_cache(version_id="v8")
    def _model_log_probs(self, prompt: str, num_choice_tokens: int) -> float:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits[:, :-1, :].squeeze(0)
            target_ids = inputs["input_ids"][:, 1:].squeeze(0)

            token_loglikelihoods = []
            for i in range(0, len(target_ids)):
                token_id = target_ids[i].item()
                token = self.tokenizer.decode([token_id])
                loglikelihood = torch.log_softmax(logits[i], dim=-1)[token_id].item()
                token_loglikelihoods.append({token: loglikelihood})

            return sum([list(log_prob.values())[0] for log_prob in token_loglikelihoods[-num_choice_tokens:]])

    @property
    def seq_length(self) -> int | None:
        config = self.model.config
        return config.max_position_embeddings if hasattr(config, "max_position_embeddings") else None


class HFLLM_from_name(HFLLM):
    """
    A generic class to create HFLLM instances from a given model name.
    """

    def __init__(self, model_name: str | None = None, formatter: str = "Llama3Formatter", **kwargs: Any) -> None:
        if model_name is None:
            raise ValueError("model_name is required")

        self.LLM_NAME = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.LLM_NAME, device_map="auto")

        # Lazy formatter initialization - only create the one we need
        selected_formatter = self._get_formatter(formatter, model_name)

        print(f"{RED}[ Model initialized --------------------- {RESET}{self.LLM_NAME} {RED}]{RESET}")
        print(f"{RED}[ Formatter: {formatter} ]{RESET}")
        self._set_formatter(selected_formatter)

    def _get_formatter(self, formatter: str, model_name: str) -> Any:
        """Get formatter instance based on formatter name."""
        if formatter == "Llama3Formatter":
            return Llama3Formatter()
        elif formatter == "MistralFormatter":
            from eval_framework.llm.mistral import MagistralFormatter

            return MagistralFormatter(model_name)
        elif formatter == "ConcatFormatter":
            return ConcatFormatter()
        elif formatter == "HFFormatter":
            return HFFormatter(model_name)
        else:
            supported = ["Llama3Formatter", "MistralFormatter", "ConcatFormatter", "HFFormatter"]
            raise ValueError(f"Unsupported formatter: {formatter}. Supported formatters: {supported}")


class Pythia410m(HFLLM):
    LLM_NAME = "EleutherAI/pythia-410m"
    DEFAULT_FORMATTER = ConcatFormatter


class SmolLM135M(HFLLM):
    LLM_NAME = "HuggingFaceTB/SmolLM-135M"
    DEFAULT_FORMATTER = ConcatFormatter


class Smollm135MInstruct(HFLLM):
    LLM_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
    DEFAULT_FORMATTER = partial(HFFormatter, LLM_NAME)


class Qwen3_0_6B(HFLLM):
    LLM_NAME = "Qwen/Qwen3-0.6B"
    DEFAULT_FORMATTER = partial(HFFormatter, LLM_NAME, chat_template_kwargs={"enable_thinking": True})
