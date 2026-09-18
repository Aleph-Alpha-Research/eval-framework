import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from dateutil import parser

from eval_framework.llm.base import BaseLLM
from eval_framework.response_generator import ResponseGenerator, repeat_samples
from eval_framework.result_processors.base import ResultProcessor
from eval_framework.result_processors.result_processor import ResultsFileProcessor
from eval_framework.shared.types import Completion, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.registry import register_task
from template_formatting.formatter import Message, Role
from tests.tests_eval_framework.conftest import MockLLM
from tests.tests_eval_framework.tasks.test_registry import temporary_registry


def test_generate_completions_message_handling() -> None:
    # Setup
    llm = Mock(spec=BaseLLM)
    config = EvalConfig(
        task_name="ARC", num_fewshot=0, num_samples=1, llm_class=llm.__class__, save_intermediate_results=False
    )
    result_processor = Mock(spec=ResultsFileProcessor)
    generator = ResponseGenerator(llm, config, result_processor)

    # Test case 1: With assistant cue message
    sample_with_cue = Sample(
        id=0,
        messages=[Message(role=Role.USER, content="Hello"), Message(role=Role.ASSISTANT, content="Cue: ")],
        ground_truth="Expected response",
        subject="no subject",
        possible_completions=None,
    )

    # Test case 2: Without assistant cue message
    sample_without_cue = Sample(
        id=1,
        messages=[Message(role=Role.USER, content="Hello")],
        ground_truth="Expected response",
        subject="no subject",
        possible_completions=None,
    )

    llm.generate.return_value = [
        RawCompletion(
            prompt="prompt",
            completion="generated text",
            prompt_num_tokens=None,
            completion_num_tokens=None,
        )
    ]
    llm.post_process_completion.side_effect = lambda completion, sample: completion

    # Execute and assert for case 1
    completion_with_cue = generator.task.generate_completions(llm, [sample_with_cue])[0]
    assert completion_with_cue.messages == [
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="Cue: generated text"),
    ]

    # Execute and assert for case 2
    completion_without_cue = generator.task.generate_completions(llm, [sample_without_cue])[0]
    assert completion_without_cue.messages == [
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="generated text"),
    ]


# test strategy:
# - expect stop sequence to be the concatenated list of llm and task stop sequences (sorted set of both)
# - expect max tokens to be the minimum of llm and task max tokens
llm_max_tokens = 999
task_max_tokens = 111
config_max_tokens = 222
llm_stop_sequences = ["stop1", "stop2"]
task_stop_sequences = ["stop3", "stop4"]
precedence_test_setup = [
    (  # llm max and nothing from task
        llm_max_tokens,
        None,
        None,
        None,
        None,
        llm_max_tokens,
        None,
    ),
    (  # task max an nothing from llm
        None,
        None,
        task_max_tokens,
        None,
        None,
        task_max_tokens,
        None,
    ),
    (  # llm max and task max
        llm_max_tokens,
        None,
        task_max_tokens,
        None,
        None,
        task_max_tokens,  # this is the smallest of the two
        None,
    ),
    (  # llm max and task max and config max
        llm_max_tokens,
        None,
        task_max_tokens,
        None,
        config_max_tokens,
        config_max_tokens,  # this is the smallest of the two and config overwrites task
        None,
    ),
    (  # llm max and task max and config max
        llm_max_tokens,
        None,
        None,
        None,
        config_max_tokens,
        config_max_tokens,  # this is the smallest of the two
        None,
    ),
    (  # llm stop and nothing from task
        None,
        llm_stop_sequences,
        None,
        None,
        None,
        None,
        llm_stop_sequences,
    ),
    (  # task stop and nothing from task
        None,
        None,
        None,
        task_stop_sequences,
        None,
        None,
        task_stop_sequences,
    ),
    (  # llm stop and task stop
        None,
        llm_stop_sequences,
        None,
        task_stop_sequences,
        None,
        None,
        list(set(llm_stop_sequences + task_stop_sequences)),
    ),
    (  # llm stop and max and nothing from task
        llm_max_tokens,
        llm_stop_sequences,
        None,
        None,
        None,
        llm_max_tokens,
        llm_stop_sequences,
    ),
    (  # task stop and max and max from llm
        llm_max_tokens,
        None,
        None,
        task_stop_sequences,
        None,
        llm_max_tokens,
        task_stop_sequences,
    ),
    (  # EVERYTHING
        llm_max_tokens,
        llm_stop_sequences,
        task_max_tokens,
        task_stop_sequences,
        config_max_tokens,
        config_max_tokens,  # this is the smallest of the two and config overwrites task
        list(set(llm_stop_sequences + task_stop_sequences)),
    ),
    (  # NOTHING
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    """
    llm_max_tokens,
    llm_stop_sequences,
    task_max_tokens,
    task_stop_sequences,
    config_max_tokens,
    expected_max_tokens,
    expected_stop_sequences
    """,
    precedence_test_setup,
)
def test_response_generator_llm_token_overloading(
    llm_max_tokens: int | None,
    llm_stop_sequences: list[str] | None,
    task_max_tokens: int | None,
    task_stop_sequences: list[str] | None,
    config_max_tokens: int | None,
    expected_max_tokens: int | None,
    expected_stop_sequences: list[str] | None,
    tmp_path: Path,
) -> None:
    """
    Test the precedence of max tokens and stop sequences in the response generator
    Max tokens and stop sequences are used with completions.
    :param llm_max_tokens: max tokens provided by llm
    :param llm_stop_sequences: stop sequence provided by llm
    :param task_max_tokens: max tokens provided by task
    :param task_stop_sequences: stop sequence provided by task
    :param config_max_tokens: max tokens provided by config
    :param expected_max_tokens: expected max tokens in the generator
    :param expected_stop_sequences: expected stop sequences in the generator
    :return: None
    """
    # setting up mock llm
    llm = MockLLM()
    # defining max_tokens and stop_sequences from parameters
    setattr(llm, "max_tokens", llm_max_tokens)
    setattr(llm, "stop_sequences", llm_stop_sequences)

    # defining task eval config
    config = EvalConfig(
        task_name="AIME2024", num_fewshot=0, num_samples=1, llm_class=llm.__class__, max_tokens=config_max_tokens
    )

    generator = ResponseGenerator(llm, config, ResultsFileProcessor(tmp_path))
    generator.task.max_tokens = task_max_tokens
    generator.task.stop_sequences = task_stop_sequences

    # no need to load from dataset
    generator.result_processor.load_responses = MagicMock(return_value=[])  # type:ignore[method-assign]

    # we don't want to write results to disk
    generator.result_processor.save_responses = MagicMock(return_value=None)  # type:ignore[method-assign]
    mock_message = [Message(role=Role.ASSISTANT, content="Hello")]

    # don't need to actually run the completion
    generator.task.generate_completions = MagicMock(  # type:ignore[method-assign]
        return_value=[
            Completion(
                id=0,
                subject="none",
                ground_truth="none",
                messages=mock_message,
                prompt="prompt",
                prompt_num_tokens=None,
                completion="completion",
                raw_completion="raw_completion",
                raw_completion_num_tokens=1,
            )
        ]
    )
    generator.task.iterate_samples = MagicMock(  # type:ignore[method-assign]
        return_value=[
            Sample(id=0, subject="none", ground_truth="none", messages=mock_message, possible_completions=None)
        ]
    )
    generated = generator.generate(lambda: False)
    # make sure that run complete is called with the precedence values
    call_kwargs = generator.task.generate_completions.call_args[1]
    called_stop_sequences = call_kwargs["stop_sequences"]
    called_max_tokens = call_kwargs["max_tokens"]

    assert generated
    assert expected_max_tokens == called_max_tokens

    expected_stop_sequences = sorted(expected_stop_sequences) if expected_stop_sequences else None
    called_stop_sequences = sorted(called_stop_sequences) if called_stop_sequences else None
    assert expected_stop_sequences == called_stop_sequences


@pytest.mark.parametrize(
    "task_subjects, expected_subjects, raises, task_name",
    [
        pytest.param(["foobar, *"], [], True, "HumanEval_OLMES", id="invalid_subjects_str"),
        pytest.param(
            ["computer_security", "conceptual_physics"],
            ["computer_security", "conceptual_physics"],
            False,
            "MMLU",
            id="valid_subjects_tuples",
        ),
        pytest.param(["computer_security", "foobar"], [], True, "MMLU", id="invalid_subjects_str"),
    ],
)
def test_filter_task_subjects(
    task_subjects: list[str], expected_subjects: list[tuple[str, str]], raises: bool, task_name: str
) -> None:
    llm = Mock(spec=BaseLLM)
    config = EvalConfig(
        task_name=task_name, num_fewshot=0, num_samples=1, task_subjects=task_subjects, llm_class=llm.__class__
    )
    result_processor = Mock(spec=ResultsFileProcessor)

    if raises:
        with pytest.raises(ValueError):
            generator = ResponseGenerator(llm, config, result_processor)
    else:
        generator = ResponseGenerator(llm, config, result_processor)
        assert sorted(generator.task.get_metadata()["subjects"]) == sorted(expected_subjects)


def test_response_generator_metadata_handling(tmp_path: Path) -> None:
    # Setup
    llm = MockLLM()
    config = EvalConfig(
        task_name="ARC", num_fewshot=0, num_samples=1, llm_class=llm.__class__, save_intermediate_results=False
    )
    config = EvalConfig(task_name="AIME2024", num_fewshot=0, num_samples=1, llm_class=llm.__class__)

    generator = ResponseGenerator(llm, config, ResultsFileProcessor(tmp_path))
    generator.generate(lambda: False)

    metadata = generator._get_metadata()
    start = parser.parse(str(metadata.get("start_time")))
    end = parser.parse(str(metadata.get("end_time")))
    total = metadata.get("total_time")
    reference = (end - start).total_seconds()

    # will fail at DST change times
    # check that clock time is before the end time
    assert start < end
    assert reference
    assert total


def test_response_generator_repeats_generates_multiple_outputs(tmp_path: Path) -> None:
    llm = MockLLM()
    config = EvalConfig(
        task_name="AIME2024",
        num_fewshot=0,
        num_samples=1,
        llm_class=MockLLM,
        repeats=3,
        batch_size=10,
        save_intermediate_results=False,
    )

    generator = ResponseGenerator(llm, config, ResultsFileProcessor(tmp_path))

    responses, preempted = generator.generate(lambda: False)
    assert len(responses) == 3
    assert all(response.prompt == responses[0].prompt for response in responses)


def test_response_generator_repeats_with_intermediate_results_writes_unique_ids(tmp_path: Path) -> None:
    llm = MockLLM()
    repeats = 3
    config = EvalConfig(
        task_name="AIME2024",
        num_fewshot=0,
        num_samples=1,
        llm_class=MockLLM,
        repeats=repeats,
        batch_size=10,
        save_intermediate_results=True,
    )

    result_processor = ResultsFileProcessor(tmp_path)
    generator = ResponseGenerator(llm, config, result_processor)

    responses, preempted = generator.generate(lambda: False)
    assert not preempted
    assert len(responses) == repeats

    # Re-loading should not treat the file as corrupted (duplicate (id, subject) pairs).
    loaded = result_processor.load_responses()
    assert len(loaded) == repeats
    assert len({(r.id, r.subject) for r in loaded}) == repeats


def test_repeat_samples() -> None:
    samples = [
        Sample(
            id=0,
            subject="foo",
            ground_truth="bar",
            messages=[Message(role=Role.USER, content="baz")],
            possible_completions=None,
        )
    ]
    repeated = list(repeat_samples(samples, 3))
    assert len(repeated) == 3
    assert [r.id for r in repeated] == [0, 1, 2]
    for other in repeated[1:]:
        assert other.subject == repeated[0].subject
        assert other.ground_truth == repeated[0].ground_truth
        assert other.messages == repeated[0].messages
        assert other.possible_completions == repeated[0].possible_completions


def test_with_wrong_loaded_metadata(tmp_path: Path) -> None:
    class OtherMockLLM(MockLLM):
        pass

    configs = [
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=1, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=1, llm_class=OtherMockLLM),
        EvalConfig(task_name="AIME2024", num_fewshot=0, num_samples=1, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=1, num_samples=1, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=2, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=1, llm_class=MockLLM, task_subjects=["ARC-Easy"]),
    ]
    configs.append(configs[0])

    # WHEN trying to run the generator with two different configs in a single output dir
    for i, config in enumerate(configs):
        generator = ResponseGenerator(config.llm_class(), config, ResultsFileProcessor(tmp_path))

        if i == 0 or i == len(configs) - 1:
            generator.generate(lambda: False)
        else:
            # THEN the second generator should raise an error because intermediate results are not compatible
            with pytest.raises(ValueError):
                generator.generate(lambda: False)


def test_generate_completions_applies_model_then_task_post_processing() -> None:
    # generate_completions must apply the model's post-processing (llm.post_process_completion) BEFORE the
    # task's own (post_process_generated_completion). The markers make the order visible:
    # raw -> MODEL[raw] -> TASK[MODEL[raw]]. This is a property of BaseTask.generate_completions, so it runs
    # on a BaseTask double built directly — no registry, no ResponseGenerator. (The composed equivalent, a
    # kind's extract_answer, is covered in the composed tests.)
    class MarkerLLM(MockLLM):
        def post_process_completion(self, completion: str, sample: Sample) -> str:
            return f"MODEL[{completion}]"

    class MarkerTask(StubTask):
        def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
            return f"TASK[{super().post_process_generated_completion(completion_text, sample)}]"

    task = MarkerTask.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    llm = MarkerLLM()
    llm.generate = Mock(  # type: ignore[method-assign]
        return_value=[
            RawCompletion(prompt="prompt", completion="raw_answer", prompt_num_tokens=None, completion_num_tokens=None)
        ]
    )
    sample = Sample(
        id=0,
        subject="stub",
        ground_truth="A",
        messages=[Message(role=Role.USER, content="Test question")],
        possible_completions=None,
    )

    completions = task.generate_completions(llm, [sample])

    assert completions[0].raw_completion == "raw_answer"
    assert completions[0].completion == "TASK[MODEL[raw_answer]]"


class SaboteurLLM(BaseLLM):
    """LLM that always raises — drives the fail_on_error code path without mocks."""

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[RawCompletion]:
        raise RuntimeError("inference connection failed")

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        raise RuntimeError("inference connection failed")


class NoopResultProcessor(ResultProcessor):
    """No-op result processor sufficient for ResponseGenerator's metadata + IO calls."""

    output_dir = Path("/tmp")

    def save_metadata(self, metadata: dict) -> None: ...
    def load_metadata(self) -> dict:
        return {}

    def save_responses(self, responses: list) -> None: ...
    def save_response(self, response: Any) -> None: ...
    def load_responses(self) -> list:
        return []

    def save_metrics_results(self, results: list) -> None: ...
    def save_metrics_result(self, result: Any) -> None: ...
    def save_aggregated_results(self, result: dict) -> None: ...
    def load_metrics_results(self) -> list:
        return []


class StubTask(BaseTask[str]):
    """Minimal task that yields `num_samples` samples (default 1), no HF dataset loader."""

    REVISION_LOCKFILE = None

    NAME = "StubTask"
    DATASET_PATH = "stub"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["stub"]
    LANGUAGE = Language.ENG
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS: list = []

    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        for i in range(num_samples or 1):
            yield Sample(
                id=i,
                messages=[Message(role=Role.USER, content=f"sample-{i}")],
                ground_truth="A",
                subject="stub",
                possible_completions=None,
            )


@temporary_registry
def test_fail_on_error_propagates_llm_failure() -> None:
    # Given a generator configured with fail_on_error=True against an LLM that always raises
    register_task(StubTask)
    config = EvalConfig(
        task_name="StubTask",
        num_fewshot=0,
        num_samples=1,
        llm_class=SaboteurLLM,
        save_intermediate_results=False,
        fail_on_error=True,
    )
    generator = ResponseGenerator(SaboteurLLM(), config, NoopResultProcessor())  # type: ignore[arg-type]

    # Then the original exception propagates
    with pytest.raises(RuntimeError, match="inference connection failed"):
        # When running the generator
        generator.generate(should_preempt_callable=lambda: False)


@temporary_registry
def test_fail_on_error_disabled_swallows_llm_failure() -> None:
    # Given a generator with the default fail_on_error=False against an LLM that always raises
    register_task(StubTask)
    config = EvalConfig(
        task_name="StubTask",
        num_fewshot=0,
        num_samples=1,
        llm_class=SaboteurLLM,
        save_intermediate_results=False,
    )
    generator = ResponseGenerator(SaboteurLLM(), config, NoopResultProcessor())  # type: ignore[arg-type]

    # When running the generator
    responses, _ = generator.generate(should_preempt_callable=lambda: False)

    # Then the failure is captured per-sample as an Error, not raised
    assert len(responses) == 1
    assert responses[0].error is not None
    assert responses[0].error.error_class == "RuntimeError"


class RecordingConcurrentLLM(BaseLLM):
    """API-style LLM: safe to call from several threads, records how it was called."""

    SUPPORTS_CONCURRENT_REQUESTS = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.call_sizes: list[int] = []
        self.started: list[str] = []

    def _on_start(self, content: str) -> None:
        pass

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[RawCompletion]:
        with self._lock:
            self.call_sizes.append(len(messages))
        completions = []
        for single_messages in messages:
            content = single_messages[0].content
            with self._lock:
                self.started.append(content)
            self._on_start(content)
            completions.append(
                RawCompletion(prompt=content, completion="ok", prompt_num_tokens=None, completion_num_tokens=None)
            )
        return completions

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        raise NotImplementedError


class GatedConcurrentLLM(RecordingConcurrentLLM):
    """The very first request only returns once a third request has started.

    With a window of 2 this can only succeed if the generator refills the slot freed by the second
    request while the first one is still running; lock-step batching would deadlock (and time out).
    """

    def __init__(self) -> None:
        super().__init__()
        self._third_started = threading.Event()

    def _on_start(self, content: str) -> None:
        with self._lock:
            if len(self.started) >= 3:
                self._third_started.set()
        if content == "sample-0":
            assert self._third_started.wait(timeout=5), "window did not slide: third request never started"


def _make_generator(llm: BaseLLM, num_samples: int, batch_size: int, **config_kwargs: Any) -> ResponseGenerator:
    register_task(StubTask)
    config = EvalConfig(
        task_name="StubTask",
        num_fewshot=0,
        num_samples=num_samples,
        llm_class=llm.__class__,
        batch_size=batch_size,
        save_intermediate_results=False,
        **config_kwargs,
    )
    return ResponseGenerator(llm, config, NoopResultProcessor())  # type: ignore[arg-type]


@temporary_registry
def test_concurrent_llm_keeps_window_full_instead_of_waiting_for_whole_batch() -> None:
    # Given an API-style LLM whose first request is slow, and a window of 2 requests
    llm = GatedConcurrentLLM()
    generator = _make_generator(llm, num_samples=3, batch_size=2)

    # When running the generator
    responses, preempted = generator.generate(should_preempt_callable=lambda: False)

    # Then the third request was started while the first was still pending, and every sample succeeded
    assert not preempted
    assert len(responses) == 3
    assert all(response.error is None for response in responses), [r.error for r in responses]
    assert sorted(response.id for response in responses) == [0, 1, 2]
    # ... each request carried a single sample rather than a whole batch
    assert llm.call_sizes == [1, 1, 1]


@temporary_registry
def test_concurrent_llm_preemption_keeps_in_flight_results_and_drops_the_rest() -> None:
    # Given a concurrent LLM, 6 samples, a window of 2, and a preemption request after the first completion
    llm = RecordingConcurrentLLM()
    generator = _make_generator(llm, num_samples=6, batch_size=2)

    # When running the generator (preemption is polled after each completed request, at which point at least
    # one request has been started)
    responses, preempted = generator.generate(should_preempt_callable=lambda: len(llm.started) > 0)

    # Then only requests from the first window are kept (the second one may have been cancelled before a worker
    # picked it up); samples beyond the window were never submitted
    assert preempted
    assert 1 <= len(responses) <= 2
    assert {response.id for response in responses} <= {0, 1}
    assert set(llm.started) <= {"sample-0", "sample-1"}


@temporary_registry
def test_concurrent_llm_fail_on_error_propagates() -> None:
    class ConcurrentSaboteurLLM(SaboteurLLM):
        SUPPORTS_CONCURRENT_REQUESTS = True

    generator = _make_generator(ConcurrentSaboteurLLM(), num_samples=3, batch_size=2, fail_on_error=True)

    with pytest.raises(RuntimeError, match="inference connection failed"):
        generator.generate(should_preempt_callable=lambda: False)


@temporary_registry
def test_concurrent_llm_fail_on_error_disabled_captures_errors_per_sample() -> None:
    class ConcurrentSaboteurLLM(SaboteurLLM):
        SUPPORTS_CONCURRENT_REQUESTS = True

    generator = _make_generator(ConcurrentSaboteurLLM(), num_samples=3, batch_size=2)

    responses, preempted = generator.generate(should_preempt_callable=lambda: False)

    assert not preempted
    assert len(responses) == 3
    assert all(response.error is not None and response.error.error_class == "RuntimeError" for response in responses)


@temporary_registry
def test_non_concurrent_llm_still_receives_whole_batches() -> None:
    # Given a local-style LLM (SUPPORTS_CONCURRENT_REQUESTS is False by default) and batch_size larger than the data
    llm = MockLLM()
    generator = _make_generator(llm, num_samples=3, batch_size=10)

    # When running the generator
    responses, preempted = generator.generate(should_preempt_callable=lambda: False)

    # Then all samples went to the model in a single call on the calling thread
    assert not preempted
    assert len(responses) == 3
    assert llm.generate_counter == 1
