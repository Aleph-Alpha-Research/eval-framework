import logging
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from typing import Any

from eval_framework.tasks.registry import get_task

try:
    from determined._info import get_cluster_info
except ImportError:
    get_cluster_info = None  # type: ignore[assignment]


from tqdm import tqdm

from eval_framework import __version__ as eval_framework_version
from eval_framework.llm.base import BaseLLM
from eval_framework.result_processors.result_processor import ResultsFileProcessor
from eval_framework.shared.types import (
    Completion,
    Error,
    Loglikelihood,
    RawCompletion,
    RawLoglikelihood,
)
from eval_framework.tasks.base import Language, ResponseType, Sample
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.perturbation import create_perturbation_class
from eval_framework.tasks.utils import raise_errors
from eval_framework.utils.constants import RED, RESET
from template_formatting.formatter import Message, Role

logger = logging.getLogger(__name__)


def map_language_to_value(
    language: Language | dict[str, Language] | dict[str, tuple[Language, Language]] | None,
) -> str | dict[str, str] | dict[str, tuple[str, str]] | None:
    if language is None:
        return None
    elif isinstance(language, Language):
        return language.value
    elif isinstance(language, dict):
        if isinstance(list(language.values())[0], Language):
            return {k: v.value for k, v in language.items()}  # type: ignore[union-attr]
        else:
            return {k: (v[0].value, v[1].value) for k, v in language.items()}  # type: ignore[index]
    else:
        raise ValueError(f"Invalid language: {language}")


class ResponseGenerator:
    def __init__(self, llm: BaseLLM, config: EvalConfig, result_processor: ResultsFileProcessor) -> None:
        self.few_shot = config.num_fewshot
        self.task_name = config.task_name
        self.llm = llm
        self.config = config
        self.result_processor = result_processor
        self.num_samples = config.num_samples
        self.save_intermediate_results = config.save_intermediate_results

        task_class = get_task(config.task_name)

        if config.perturbation_config is not None:
            perturbation_task_class = create_perturbation_class(task_class, config.perturbation_config)
            self.task = perturbation_task_class.with_overwrite(
                self.few_shot, custom_subjects=self.config.task_subjects, custom_hf_revision=self.config.hf_revision
            )
        else:
            self.task = task_class.with_overwrite(
                self.few_shot, custom_subjects=self.config.task_subjects, custom_hf_revision=self.config.hf_revision
            )

        self.response_type = task_class.RESPONSE_TYPE

    def _llm_task_param_precedence(self) -> tuple[list[str] | None, int | None]:
        """
        sets the stop_sequences and max_tokens values to be used in the completion generation.
        Max token and stop sequence values have an order of precedence:

        LLM attributes take precedence over task attributes, and therefore overload them.
        :return: stop_sequences, max_tokens
        """
        llm_stop_sequences = getattr(self.llm, "stop_sequences", None)
        llm_max_tokens = getattr(self.llm, "max_tokens", None)
        task_stop_sequences = getattr(self.task, "stop_sequences", None)
        task_max_tokens = self.config.max_tokens or getattr(self.task, "max_tokens", None)
        # if both task and model define a max_token, the smaller value is used
        max_tokens = min([x for x in [llm_max_tokens, task_max_tokens] if x is not None], default=None)
        logger.info(f"Set max_tokens to {max_tokens}")
        # if both task and model define stop sequences, those are merged into one list
        stop_sequences_merged = (llm_stop_sequences or []) + (task_stop_sequences or [])
        stop_sequences = sorted(list(set(stop_sequences_merged))) if stop_sequences_merged else None
        logger.info(f"Set stop_sequences to {stop_sequences}")
        return stop_sequences, max_tokens

    def _generate_completions(
        self,
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> list[Completion]:
        """
        Generates completions for the sample.
        :param sample: sample to generate completions for
        :param stop_sequences: stop sequences to use in completion generation
        :param max_tokens: maximum tokens to use in completion generation
        :return: completion
        """
        if stop_sequences is None:
            stop_sequences = []

        raw_completions: list[RawCompletion]
        try:
            raw_completions = self.llm.generate(samples=samples, stop_sequences=stop_sequences, max_tokens=max_tokens)
        except Exception as e:
            if raise_errors():
                raise e
            logger.info(f"Error: {e.__class__.__name__} {e}")
            assert len(samples) == 1, "LLMs not handling errors are not supported in batch mode"
            raw_completions = [
                RawCompletion(
                    prompt="",
                    prompt_sequence_positions=0,
                    completion="",
                    completion_sequence_positions=0,
                    raw_completion_error=Error(
                        error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc()
                    ),
                )
                for _ in range(len(samples))
            ]

        completion_list = []
        for idx, sample in enumerate(samples):
            raw_completion = raw_completions[idx]

            if sample.messages and sample.messages[-1].role == Role.ASSISTANT:
                messages = sample.messages[:-1] + [
                    Message(role=Role.ASSISTANT, content=sample.messages[-1].content + raw_completion.completion)
                ]
            else:
                messages = sample.messages + [Message(role=Role.ASSISTANT, content=raw_completion.completion)]

            try:
                error = None
                completion = self.task.post_process_generated_completion(raw_completion.completion, sample)
            except Exception as e:
                error = Error(error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc())
                completion = ""

            completion_list.append(
                Completion(
                    id=sample.id,
                    subject=sample.subject,
                    ground_truth=sample.ground_truth,
                    prompt=raw_completion.prompt,
                    prompt_sequence_positions=raw_completion.prompt_sequence_positions,
                    concat_compression=raw_completion.concat_compression,
                    messages=messages,
                    completion=completion,
                    raw_completion=raw_completion.completion,
                    raw_completion_sequence_positions=raw_completion.completion_sequence_positions,
                    context=sample.context,
                    error=raw_completion.raw_completion_error or error,
                )
            )

        return completion_list

    def _generate_loglikelihoods(self, samples: list[Sample]) -> list[Loglikelihood]:
        """
        Generate log likelihoods when a sample is run against the model.
        :param sample: sample to run the task against
        :return: loglikelihoods
        """
        raw_loglikelihoods: list[RawLoglikelihood]
        try:
            raw_loglikelihoods = self.llm.logprobs(samples)
        except Exception as e:
            if raise_errors():
                raise e
            logger.info(f"Error: {e.__class__.__name__} {e}")
            assert len(samples) == 1, "LLMs not handling errors are not supported in batch mode"
            raw_loglikelihoods = [
                RawLoglikelihood(
                    prompt="",
                    prompt_sequence_positions=0,
                    loglikelihoods={},
                    loglikelihoods_sequence_positions={},
                    raw_loglikelihood_error=Error(
                        error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc()
                    ),
                )
                for _ in range(len(samples))
            ]

        loglikelihood_list = []
        for idx, sample in enumerate(samples):
            raw_loglikelihood = raw_loglikelihoods[idx]
            assert sample.ground_truth is not None
            loglikelihood_list.append(
                Loglikelihood(
                    id=sample.id,
                    subject=sample.subject,
                    ground_truth=sample.ground_truth,
                    prompt=raw_loglikelihood.prompt,
                    prompt_sequence_positions=raw_loglikelihood.prompt_sequence_positions,
                    concat_compression=raw_loglikelihood.concat_compression,
                    loglikelihoods=raw_loglikelihood.loglikelihoods,
                    loglikelihoods_sequence_positions=raw_loglikelihood.loglikelihoods_sequence_positions,
                    error=raw_loglikelihood.raw_loglikelihood_error,
                )
            )
        return loglikelihood_list

    def _generative_output_type_selector(self) -> Callable[[list[Sample]], list[Completion] | list[Loglikelihood]]:
        """
        Selects the generative output type based on the response type.
        :return: function to generate responses
        """
        match self.response_type:
            case ResponseType.COMPLETION:
                stop_sequences, max_tokens = self._llm_task_param_precedence()
                return partial(self._generate_completions, stop_sequences=stop_sequences, max_tokens=max_tokens)  # type: ignore[call-arg]
            case ResponseType.LOGLIKELIHOODS:
                return self._generate_loglikelihoods
            case _:
                raise KeyError(f"Task type {self.task} not supported")

    def _run_task_against_model(
        self, should_preempt_callable: Callable[[], bool]
    ) -> tuple[list[Completion | Loglikelihood], bool]:
        """
        Runs the task against the model and generates responses.
        :param should_preempt_callable: function to check if preempt is called
        :return: list of responses, preempted
        """
        logger.info(f"{RED}[ Running task {self.task.NAME} against model ------------ ]{RESET}")
        self.start_time, monotonic_start = time.time(), time.monotonic()
        run_fn = self._generative_output_type_selector()
        self._verify_loaded_metadata_compatibility()
        responses = self.result_processor.load_responses()  # load responses if present
        subject_response_id_mapping = self._map_subject_response_ids(responses)
        self.result_processor.save_metadata(self._get_metadata())
        responses, preempted = self._curate_responses(
            responses, subject_response_id_mapping, run_fn, should_preempt_callable
        )
        self.end_time, monotonic_end = time.time(), time.monotonic()
        self.total_time = monotonic_end - monotonic_start
        self.result_processor.save_metadata(self._get_metadata())  # overwrite with updated timing

        return responses, preempted

    def _map_subject_response_ids(self, responses: list[Completion | Loglikelihood]) -> dict[str, set[int]]:
        """
        Maps subject to response id
        :param responses: list of responses
        :return: mapping of subject to response id
        """
        subject_response_id_mapping = {}
        if responses:
            response_subjects = {resp.subject for resp in responses}
            subject_response_id_mapping = {
                response_subject: set([resp.id for resp in responses if resp.subject == response_subject])
                for response_subject in response_subjects
            }

        return subject_response_id_mapping

    def _curate_responses(
        self,
        responses: list[Completion | Loglikelihood],
        subject_response_id_mapping: dict[str, set[int]],
        generative_output_function: Callable[[list[Sample]], list[Completion] | list[Loglikelihood]],
        should_preempt_callable: Callable[[], bool],
    ) -> tuple[list[Completion | Loglikelihood], bool]:
        """
        Generates responses for the task and saves them along with metadata.
        :param responses: list of responses
        :param subject_response_id_mapping: mapping of subject to response id
        :param generative_output_function: function to generate responses
        :param metadata: metadata dictionary
        :param should_preempt_callable: function to check if preempt is called
        :return: None
        """

        def _process_batch(samples_batch: list[Sample]) -> None:
            if not samples_batch:
                return
            if len(samples_batch) > 1:
                log_msg = "Processing batch..."
                logger.info(log_msg)  # For log files
                tqdm.write(log_msg)  # For console display with tqdm

            responses_batch = generative_output_function(samples_batch)
            responses.extend(responses_batch)
            if self.save_intermediate_results:
                for response in responses_batch:
                    self.result_processor.save_response(response)

        # In order to enable parallelism we group samples in batches and send them in parallel to the `run_fn`.
        # The BaseLLM class is then in charge of managing the parallelism (eg, using AsyncClient in API models).
        # If samples_batch_size = 1, samples are run sequentially; in any case, we return here after finishing each
        # individual batch to honor preemption requests and save cached results.
        samples_batch_size = self.config.batch_size

        # Calculate total samples for progress bar - use num_samples or iterate to count
        total_num_samples = self.num_samples
        if total_num_samples is None:
            # Count samples by iterating (this might be expensive for large datasets)
            total_num_samples = sum(1 for _ in self.task.iterate_samples(None))

        samples_batch: list[Sample] = []
        with tqdm(total=total_num_samples, desc=f"Processing {self.response_type.value}") as pbar:
            for i, sample in enumerate(self.task.iterate_samples(self.num_samples)):
                subject = f" - Subject: {sample.subject}"
                sample_index = i + 1

                if sample.id in subject_response_id_mapping.get(sample.subject, []):
                    log_msg = (
                        f"Task: {self.response_type.value}{subject} - Sample: {sample_index} - skipping, already done."
                    )
                    logger.info(log_msg)  # For log files
                    tqdm.write(log_msg)  # For console display with tqdm
                    pbar.update(1)
                    continue

                log_msg = f"Task: {self.response_type.value}{subject} - Sample: {sample_index}/{total_num_samples}"
                logger.info(log_msg)  # For log files
                tqdm.write(log_msg)  # For console display with tqdm
                pbar.set_postfix_str(f"Sample {sample_index}/{total_num_samples}")
                pbar.update(1)

                samples_batch.append(sample)

                if len(samples_batch) >= samples_batch_size:
                    _process_batch(samples_batch)
                    samples_batch = []

                if should_preempt_callable():
                    log_msg = "Preempt"
                    logger.info(log_msg)  # For log files
                    tqdm.write(log_msg)  # For console display with tqdm
                    if not self.save_intermediate_results:
                        self.result_processor.save_responses(responses)
                    return responses, True

            _process_batch(samples_batch)

        if not self.save_intermediate_results:
            self.result_processor.save_responses(responses)
        return responses, False

    def _get_metadata(self) -> dict[str, Any]:
        """Prepares metadata dictionary from the configuration."""
        all_metrics = getattr(self.task, "METRICS", None)
        metadata = self.config.model_dump()
        metadata["llm_name"] = self.llm.name
        metadata["task_name"] = self.task_name
        language = getattr(self.task, "LANGUAGE", None)
        metadata["language"] = map_language_to_value(language)
        metadata["metrics"] = [m.NAME for m in all_metrics] if all_metrics is not None else []
        metadata["primary_metrics"] = getattr(self.task, "PRIMARY_METRICS", None)
        metadata["eval_framework_version"] = eval_framework_version
        metadata["task_output_dir"] = str(self.result_processor.output_dir)
        if hasattr(self, "total_time"):
            metadata["start_time"] = str(datetime.fromtimestamp(self.start_time, UTC))
            metadata["end_time"] = str(datetime.fromtimestamp(self.end_time, UTC))
            metadata["total_time"] = self.total_time

        # add task specific metadata
        metadata["task_metadata"] = self.task.get_metadata()
        
        try:
            assert get_cluster_info is not None, "Determined cluster info not available"
            info = get_cluster_info()
            if info is not None:
                metadata["determined_agent_id"] = info.agent_id
                if info.task_type == "TRIAL":
                    metadata["determined_experiment_id"] = info.trial.experiment_id
                    metadata["determined_trial_id"] = info.trial.trial_id
        except Exception as e:
            logger.info(f"{e}; cluster info not available in local context")

        return metadata

    def _verify_loaded_metadata_compatibility(self) -> None:
        if not (loaded_metadata := self.result_processor.load_metadata()):
            return
        current_metadata = self._get_metadata()
        # check if crucial keys in metadata are the same as in the previous run
        keys = [
            "task_name",
            "task_subjects",
            "num_fewshot",
            "num_samples",
            "llm_name",
            "llm_args",
            "perturbation_config",
        ]
        for key in keys:
            if loaded_metadata[key] != current_metadata[key]:
                raise ValueError(f"Existing metadata does not match current metadata for {key}.")

    def generate(self, should_preempt_callable: Callable[[], bool]) -> tuple[list[Completion | Loglikelihood], bool]:
        """Generates responses and saves them along with metadata.
        :param should_preempt_callable: function to check if preempt is called
        :return: list of responses, preempted: whether the process was preempted or not
        """
        logger.info(f"{RED}[ Running responses generation ---------- ]{RESET}")
        logger.info(f"{RED}[ Will save into {self.result_processor.output_dir} ---------- ]{RESET}")
        responses, preempted = self._run_task_against_model(should_preempt_callable)
        logger.info("Completions generated and saved.")

        return responses, preempted
