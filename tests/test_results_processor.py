import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from unittest.mock import patch

from eval_framework.llm.huggingface import Qwen3_0_6B
from eval_framework.result_processors.base import Result
from eval_framework.result_processors.result_processor import ResultsFileProcessor, generate_output_dir
from eval_framework.shared.types import BaseMetricContext, Completion, Loglikelihood
from eval_framework.tasks.benchmarks.mmlu import MMLU
from eval_framework.tasks.eval_config import EvalConfig
from template_formatting.formatter import Message, Role

# here no tests are listed that would need a GPU runner -> no pytest markers set


def test_generate_output_dir_with_valid_values() -> None:
    llm_name = "llama-3.1"
    task_name = MMLU.NAME
    config = EvalConfig(
        output_dir=Path(__file__).parent / "eval_framework_results",
        num_fewshot=5,
        num_samples=10,
        task_name=task_name,
        llm_class=Qwen3_0_6B,
    )

    version_str = f"v{importlib.metadata.version('eval_framework')}"

    timestamp_mock = "20231013T120000"
    with patch("eval_framework.result_processors.result_processor.datetime") as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = timestamp_mock
        output_dir = generate_output_dir(llm_name, config)

    fewshot_str = f"fewshot_{config.num_fewshot}"
    samples_str = f"samples_{config.num_samples}"
    params_str = f"{fewshot_str}__{samples_str}"
    config_json = config.model_json_robust_subset_dump()
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:5]

    dir_name = f"{params_str}_{config_hash}"

    expected_path = Path(config.output_dir) / llm_name / f"{version_str}_{task_name}" / dir_name
    assert output_dir == expected_path


def test_generate_output_dir_with_none_values() -> None:
    llm_name = "llama-3.1"
    task_name = MMLU.NAME
    config = EvalConfig(
        output_dir=Path("/eval_framework_results"),
        num_fewshot=0,
        num_samples=None,
        task_name=task_name,
        llm_class=Qwen3_0_6B,
    )

    version_str = f"v{importlib.metadata.version('eval_framework')}"

    timestamp_mock = "20231013T120000"
    with patch("eval_framework.result_processors.result_processor.datetime") as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = timestamp_mock
        output_dir = generate_output_dir(llm_name, config)

    fewshot_str = "fewshot_0"
    samples_str = "samples_None"
    params_str = f"{fewshot_str}__{samples_str}"
    config_json = config.model_json_robust_subset_dump()
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:5]
    dir_name = f"{params_str}_{config_hash}"

    expected_path = Path(config.output_dir) / llm_name / f"{version_str}_{task_name}" / dir_name
    assert output_dir == expected_path


def test_file_result_processor_save_and_load_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "test_output"
    processor = ResultsFileProcessor(output_dir)

    responses: list[Completion | Loglikelihood] = []
    responses.append(
        Completion(
            id=1,
            subject="math",
            ground_truth="4",
            prompt="What is 2+2?",
            prompt_sequence_positions=None,
            messages=[Message(role=Role.USER, content="Sample prompt 1")],
            completion="4",
            raw_completion="4",
            raw_completion_sequence_positions=None,
        )
    )
    responses.append(
        Loglikelihood(
            id=2,
            subject="math",
            prompt="What is 2+2?",
            prompt_sequence_positions=None,
            ground_truth="4",
            loglikelihoods={"1": -2.0, "2": -1.5, "3": -1.2, "4": -0.5},
            loglikelihoods_sequence_positions={"1": -1, "2": -1, "3": -1, "4": -1},
        )
    )
    metadata = {"llm_name": "TestLLM", "task_name": "TestTask", "num_fewshot": 0, "num_samples": 10}

    output_file = output_dir / "output.jsonl"
    metadata_file = output_dir / "metadata.json"

    for response in responses:
        processor.save_response(response)
    assert output_file.exists(), "Output file was not created."
    loaded_responses = processor.load_responses()
    for a, b in zip(responses, loaded_responses, strict=True):
        assert a.model_dump() == b.model_dump()

    output_file.unlink()

    processor.save_responses(responses)
    loaded_responses = processor.load_responses()
    for a, b in zip(responses, loaded_responses, strict=True):
        assert a.model_dump() == b.model_dump()

    processor.save_metadata(metadata)

    assert metadata_file.exists(), "Metadata file was not created."
    loaded_metadata = processor.load_metadata()
    assert loaded_metadata == metadata


def test_file_result_processor_load_invalid_output(tmp_path: Path) -> None:
    # GIVEN an invalid jsonl file
    with open(tmp_path / "output.jsonl", "w") as f:
        f.write("This is not a valid jsonl file.")

    # THEN results should not be loaded
    processor = ResultsFileProcessor(tmp_path)
    loaded = processor.load_responses()
    assert loaded == []
    assert len(list(tmp_path.glob("output.jsonl.broken.*"))) == 1


def test_file_result_processor_load_duplicate_output(tmp_path: Path) -> None:
    # GIVEN a valid jsonl file with duplicate IDs
    completion = Completion(
        id=123,
        subject="math",
        ground_truth="4",
        prompt="What is 2+2?",
        prompt_sequence_positions=None,
        messages=[Message(role=Role.USER, content="Sample prompt 1")],
        completion="4",
        raw_completion="4",
        raw_completion_sequence_positions=None,
    )
    processor = ResultsFileProcessor(tmp_path)
    processor.save_responses([completion, completion])

    # THEN results should not be loaded
    loaded = processor.load_responses()
    assert loaded == []
    assert len(list(tmp_path.glob("output.jsonl.broken.*"))) == 1


def test_file_result_processor_save_and_load_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "test_output"
    processor = ResultsFileProcessor(output_dir)

    results = []
    for i in range(2):
        results.append(
            Result(
                id=i,
                subject="math",
                num_fewshot=2,
                llm_name="TestLLM",
                task_name="TestTask",
                metric_class_name="TestMetric",
                metric_name="TestMetric",
                key=None,
                value=0.5,
                higher_is_better=True,
                prompt="What is 2+2?",
                response="4",
            )
        )

    output_file = output_dir / "results.jsonl"

    for result in results:
        processor.save_metrics_result(result)
    assert output_file.exists(), "Output file was not created."
    loaded_results = processor.load_metrics_results()
    for a, b in zip(results, loaded_results, strict=True):
        assert a.model_dump() == b.model_dump()

    output_file.unlink()

    processor.save_metrics_results(results)
    loaded_results = processor.load_metrics_results()
    for a, b in zip(results, loaded_results, strict=True):
        assert a.model_dump() == b.model_dump()


def test_file_result_processor_custom_context(tmp_path: Path) -> None:
    """Test that custom BaseMetricContext subclasses are properly saved and loaded."""

    # Create a custom BaseMetricContext subclass with custom fields
    class CustomContext(BaseMetricContext):
        foo: str = "bar"
        count: int = 42
        nested: dict = {"key": "value"}

    # Create a test output directory
    output_dir = tmp_path / "test_custom_context"
    processor = ResultsFileProcessor(output_dir)

    # Create a Completion with our custom context
    custom_context = CustomContext()
    completion = Completion(
        id=999,
        subject="custom_context_test",
        ground_truth="test",
        prompt="Test with custom context",
        prompt_sequence_positions=None,
        messages=[Message(role=Role.USER, content="Sample prompt")],
        completion="test response",
        raw_completion="test response",
        raw_completion_sequence_positions=None,
        context=custom_context,  # Use our custom context
    )

    # Save the completion
    processor.save_response(completion)

    # Load the saved completion
    loaded_responses = processor.load_responses()
    assert len(loaded_responses) == 1

    # Check that the custom context was properly loaded
    loaded_completion = loaded_responses[0]
    assert isinstance(loaded_completion, Completion), "Expected Completion object"
    assert loaded_completion.context is not None
    assert isinstance(loaded_completion.context, BaseMetricContext)

    # Check custom fields
    assert hasattr(loaded_completion.context, "foo")
    assert loaded_completion.context.foo == "bar"
    assert hasattr(loaded_completion.context, "count")
    assert loaded_completion.context.count == 42
    assert hasattr(loaded_completion.context, "nested")
    assert loaded_completion.context.nested == {"key": "value"}


def test_file_result_processor_list_of_custom_contexts(tmp_path: Path) -> None:
    """Test that lists of custom BaseMetricContext objects are properly saved and loaded."""

    # Create a custom BaseMetricContext subclasses with custom fields
    class CustomContext1(BaseMetricContext):
        name: str = "context1"
        data: list = ["a", "b", "c"]

    class CustomContext2(BaseMetricContext):
        name: str = "context2"
        scores: dict = {"precision": 0.95, "recall": 0.87}

    # Create a test output directory
    output_dir = tmp_path / "test_multiple_custom_contexts"
    processor = ResultsFileProcessor(output_dir)

    # Create a list of contexts
    context_list = [CustomContext1(), CustomContext2()]

    # Create a Completion with our list of contexts
    completion = Completion(
        id=888,
        subject="multiple_contexts_test",
        ground_truth="test",
        prompt="Test with multiple custom contexts",
        prompt_sequence_positions=None,
        messages=[Message(role=Role.USER, content="Sample prompt")],
        completion="test response",
        raw_completion="test response",
        raw_completion_sequence_positions=None,
        context=context_list,  # Use our list of contexts
    )

    # Save the completion
    processor.save_response(completion)

    # Load the saved completion
    loaded_responses = processor.load_responses()
    assert len(loaded_responses) == 1

    # Check that the contexts were properly loaded
    loaded_completion = loaded_responses[0]
    assert isinstance(loaded_completion, Completion), "Expected Completion object"
    assert loaded_completion.context is not None
    assert isinstance(loaded_completion.context, list)
    assert len(loaded_completion.context) == 2

    # Check first context
    context1 = loaded_completion.context[0]
    assert hasattr(context1, "name")
    assert context1.name == "context1"
    assert hasattr(context1, "data")
    assert context1.data == ["a", "b", "c"]

    # Check second context
    context2 = loaded_completion.context[1]
    assert hasattr(context2, "name")
    assert context2.name == "context2"
    assert hasattr(context2, "scores")
    assert context2.scores == {"precision": 0.95, "recall": 0.87}


def test_file_result_processor_batch_custom_contexts(tmp_path: Path) -> None:
    """Test that batch saving of completions with custom contexts works properly."""

    # Create custom context classes
    class MetricsContext(BaseMetricContext):
        metrics: dict = {"accuracy": 0.92, "f1": 0.88}

    class DebugContext(BaseMetricContext):
        debug_info: dict = {"runtime": 1.25, "memory": 512}
        logs: list = ["info: started", "debug: processing", "info: completed"]

    # Create a test output directory
    output_dir = tmp_path / "test_batch_custom_contexts"
    processor = ResultsFileProcessor(output_dir)

    # Create completions with custom contexts
    completion1 = Completion(
        id=777,
        subject="batch_test_1",
        ground_truth="answer1",
        prompt="Test prompt 1",
        prompt_sequence_positions=None,
        messages=[Message(role=Role.USER, content="Prompt 1")],
        completion="answer1",
        raw_completion="answer1",
        raw_completion_sequence_positions=None,
        context=MetricsContext(),
    )

    completion2 = Completion(
        id=778,
        subject="batch_test_2",
        ground_truth="answer2",
        prompt="Test prompt 2",
        prompt_sequence_positions=None,
        messages=[Message(role=Role.USER, content="Prompt 2")],
        completion="answer2",
        raw_completion="answer2",
        raw_completion_sequence_positions=None,
        context=DebugContext(),
    )

    # Save completions in batch
    processor.save_responses([completion1, completion2])

    # Load the saved completions
    loaded_responses = processor.load_responses()
    assert len(loaded_responses) == 2

    # Check first completion's context
    completion1_loaded = next(c for c in loaded_responses if c.id == 777)
    assert isinstance(completion1_loaded, Completion), "Expected Completion object"
    assert completion1_loaded.context is not None
    assert isinstance(completion1_loaded.context, BaseMetricContext), "Expected BaseMetricContext"

    # check for the right content of completion1_loaded
    assert hasattr(completion1_loaded.context, "metrics")
    assert completion1_loaded.context.metrics == {"accuracy": 0.92, "f1": 0.88}

    # Check second completion's context
    completion2_loaded = next(c for c in loaded_responses if c.id == 778)
    assert isinstance(completion2_loaded, Completion), "Expected Completion object"
    assert completion2_loaded.context is not None
    assert isinstance(completion2_loaded.context, BaseMetricContext), "Expected BaseMetricContext"

    # check for the right content of completion2_loaded
    assert hasattr(completion2_loaded.context, "debug_info")
    assert completion2_loaded.context.debug_info == {"runtime": 1.25, "memory": 512}
    assert hasattr(completion2_loaded.context, "logs")
    assert completion2_loaded.context.logs == ["info: started", "debug: processing", "info: completed"]
