from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from dateutil import parser
from huggingface_hub.errors import RevisionNotFoundError

from eval_framework.llm.base import BaseLLM
from eval_framework.response_generator import ResponseGenerator
from eval_framework.result_processors.result_processor import ResultsFileProcessor
from eval_framework.shared.types import Completion, RawCompletion
from eval_framework.tasks.base import Sample
from eval_framework.tasks.benchmarks.arc import ARC
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.perturbation import PerturbationConfig, PerturbationType
from eval_framework.tasks.registry import get_task
from template_formatting.formatter import Message, Role
from tests.tests_eval_framework.conftest import MockLLM


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
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
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
                prompt_sequence_positions=None,
                completion="completion",
                raw_completion="raw_completion",
                raw_completion_sequence_positions=1,
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
    called_stop_sequences, called_max_tokens = generator.task.generate_completions.call_args[1].values()

    assert generated
    assert expected_max_tokens == called_max_tokens

    expected_stop_sequences = sorted(expected_stop_sequences) if expected_stop_sequences else None
    called_stop_sequences = sorted(called_stop_sequences) if called_stop_sequences else None
    assert expected_stop_sequences == called_stop_sequences


@pytest.mark.parametrize(
    "task_subjects, expected_subjects, raises, task_name",
    [
        pytest.param(
            # ["DP, *", "*, DataAnalysis", "PoT, FactChecking"],
            ["TCoT, *", "*, DataAnalysis", "PoT, FactChecking"],
            [
                # ("DP", "NumericalReasoning"),
                # ("DP", "DataAnalysis"),
                # ("DP", "FactChecking"),
                ("PoT", "DataAnalysis"),
                ("PoT", "FactChecking"),
                ("SCoT", "DataAnalysis"),
                ("TCoT", "NumericalReasoning"),
                ("TCoT", "DataAnalysis"),
                ("TCoT", "FactChecking"),
            ],
            False,
            "TableBench",
            id="valid_subjects_tuples",
        ),
        pytest.param(["foobar, *"], [], True, "TableBench", id="invalid_subjects_str"),
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
        with pytest.raises(AssertionError):
            generator = ResponseGenerator(llm, config, result_processor)
    else:
        generator = ResponseGenerator(llm, config, result_processor)
        assert sorted(generator.task.SUBJECTS) == sorted(expected_subjects)


@pytest.mark.parametrize(
    "task_name, hf_revision, raises",
    [
        pytest.param("Math", None, False),
        pytest.param("Math", "not_valid_revision_1", True),
        pytest.param("ARC", None, False),
        pytest.param("IFEval", "9381f5d15347ba8854ffa2a480984ce7e554ef56", False),  # old valid revision
    ],
)
def test_hf_revisions(task_name: str, hf_revision: str, raises: bool) -> None:
    llm = Mock(spec=BaseLLM)
    config = EvalConfig(
        task_name=task_name, num_fewshot=0, num_samples=1, hf_revision=hf_revision, llm_class=llm.__class__
    )
    result_processor = Mock(spec=ResultsFileProcessor)
    response_generator = ResponseGenerator(
        llm=llm,
        config=config,
        result_processor=result_processor,
    )

    if raises:
        with pytest.raises(RevisionNotFoundError):
            for _ in response_generator.task.iterate_samples(num_samples=config.num_samples):
                pass
    else:
        for _ in response_generator.task.iterate_samples(num_samples=config.num_samples):
            pass
        assert response_generator.task.dataset


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


@patch("eval_framework.response_generator.create_perturbation_class")
def test_with_wrong_loaded_metadata(mock_create_perturbation_class: Mock, tmp_path: Path) -> None:
    class OtherMockLLM(MockLLM):
        pass

    configs = [
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=1, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=1, llm_class=OtherMockLLM),
        EvalConfig(task_name="AIME2024", num_fewshot=0, num_samples=1, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=1, num_samples=1, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=2, llm_class=MockLLM),
        EvalConfig(task_name="ARC", num_fewshot=0, num_samples=1, llm_class=MockLLM, task_subjects=["ARC-Easy"]),
        EvalConfig(
            task_name="ARC", num_fewshot=0, num_samples=1, llm_class=MockLLM, perturbation_config=PerturbationConfig()
        ),
    ]
    configs.append(configs[0])

    # WHEN trying to run the generator with two different configs in a single output dir
    for i, config in enumerate(configs):
        mock_create_perturbation_class.side_effect = lambda x, _: x  # don't spin up docker here just for the test
        generator = ResponseGenerator(config.llm_class(), config, ResultsFileProcessor(tmp_path))

        if i == 0 or i == len(configs) - 1:
            generator.generate(lambda: False)
        else:
            # THEN the second generator should raise an error because intermediate results are not compatible
            with pytest.raises(ValueError):
                generator.generate(lambda: False)


@pytest.mark.gpu  # default CI worker can't handle large docker images
@pytest.mark.parametrize("perturbation_type", list(PerturbationType))
def test_perturbed_response_differs(tmp_path: Path, perturbation_type: PerturbationType) -> None:
    """Test that perturbed responses differ from original samples for each perturbation type."""
    output_dir = tmp_path / "eval"
    perturbed_eval_config = EvalConfig(
        task_name=ARC.NAME,  # Use a simple task for testing
        num_fewshot=0,
        num_samples=1,
        output_dir=output_dir,
        llm_class=MockLLM,
        llm_judge_class=MockLLM,
        judge_model_args={},
        perturbation_config=PerturbationConfig(
            type=perturbation_type,
            probability=1.0,  # Always perturb
            verbose=True,
        ),
        save_intermediate_results=False,
    )

    task_class = get_task(perturbed_eval_config.task_name)
    task = task_class()
    perturbed_response_generator = ResponseGenerator(MockLLM(), perturbed_eval_config, Mock(spec=ResultsFileProcessor))

    assert len(task.SUBJECTS) > 0
    task._load_dataset(task.SUBJECTS[0])
    original_item = task.dataset[task.SAMPLE_SPLIT][0]
    original_sample = task._get_instruction_text(original_item)
    perturbed_sample = perturbed_response_generator.task._get_instruction_text(original_item)
    assert original_sample != perturbed_sample, (
        f"Original sample should differ from perturbed sample for perturbation type {perturbation_type}"
    )


def test_response_generator_applies_model_then_task_post_processing(tmp_path: Path) -> None:
    class MarkerLLM(MockLLM):
        def post_process_completion(self, completion: str, sample: Sample) -> str:
            return f"MODEL[{completion}]"

    llm = MarkerLLM()
    config = EvalConfig(
        task_name="ARC",
        num_fewshot=0,
        num_samples=1,
        llm_class=llm.__class__,
        save_intermediate_results=False,
    )
    result_processor = ResultsFileProcessor(tmp_path)
    generator = ResponseGenerator(llm, config, result_processor)

    original_task_post_process = generator.task.post_process_generated_completion

    def task_post_process_with_marker(completion: str, sample: Sample | None = None) -> str:
        result = original_task_post_process(completion, sample)
        return f"TASK[{result}]"

    generator.task.post_process_generated_completion = task_post_process_with_marker  # type: ignore[method-assign, assignment]

    sample = Sample(
        id=0,
        subject="ARC-Easy",
        ground_truth="A",
        messages=[Message(role=Role.USER, content="Test question")],
        possible_completions=None,
    )

    llm.generate = Mock(  # type: ignore[method-assign]
        return_value=[
            RawCompletion(
                prompt="prompt",
                completion="raw_answer",
                prompt_sequence_positions=None,
                completion_sequence_positions=None,
            )
        ]
    )

    completions = generator.task.generate_completions(llm, [sample])

    assert completions[0].raw_completion == "raw_answer"
    assert completions[0].completion == "TASK[MODEL[raw_answer]]"
