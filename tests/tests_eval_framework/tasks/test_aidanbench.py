from unittest.mock import Mock, patch

from eval_framework.metrics.completion.aidanbench import AidanBenchMetric
from eval_framework.shared.types import Completion
from eval_framework.tasks.base import Sample
from eval_framework.tasks.benchmarks.aidanbench import COHERENCE_THRESHOLD, AidanBench
from template_formatting.formatter import Message, Role


def test_get_instruction_text():
    """Test that instruction text is formatted correctly."""
    task = AidanBench(num_fewshot=0)
    item = {"prompt": "What is the capital of France?"}

    instruction = task._get_instruction_text(item)

    assert "What is the capital of France?" in instruction
    assert "<question>" in instruction
    assert "</question>" in instruction
    assert "<answer>" in instruction
    assert "</answer>" in instruction
    assert "one direct answer" in instruction
    assert "DO NOT list multiple answers" in instruction


def test_get_ground_truth():
    """Test that ground truth is None for AidanBench."""
    task = AidanBench(num_fewshot=0)
    item = {"prompt": "What is the capital of France?"}

    ground_truth = task._get_ground_truth(item)

    assert ground_truth is None


def test_calculate_novelty_score_single_response():
    """Test novelty calculation with single response."""
    task = AidanBench(num_fewshot=0)

    messages = [
        Message(role=Role.USER, content="What is the capital of France?"),
        Message(role=Role.ASSISTANT, content="Paris is the capital of France."),
    ]

    novelty = task._calculate_novelty_score(messages)

    assert novelty == 1.0  # Single response is maximally novel


@patch("eval_framework.tasks.benchmarks.aidanbench.pairwise_cosine_similarity")
def test_calculate_novelty_score_multiple_responses(mock_cosine_similarity):
    """Test novelty calculation with multiple responses."""
    task = AidanBench(num_fewshot=0)

    # Mock the embedding model
    mock_embedding_model = Mock()
    mock_embedding_model.generate_from_messages.return_value = [
        [0.1, 0.2, 0.3],  # First response embedding
        [0.4, 0.5, 0.6],  # Second response embedding
    ]
    task._embedding_model = mock_embedding_model

    # Mock cosine similarity to return high similarity (0.8)
    mock_cosine_similarity.return_value = [[0.8]]

    messages = [
        Message(role=Role.USER, content="What is the capital of France?"),
        Message(role=Role.ASSISTANT, content="Paris is the capital."),
        Message(role=Role.ASSISTANT, content="The capital is Paris."),
    ]

    novelty = task._calculate_novelty_score(messages)

    # Novelty should be 1 - max_similarity = 1 - 0.8 = 0.2
    assert abs(novelty - 0.2) < 1e-10

    # Verify the embedding model was called correctly
    mock_embedding_model.generate_from_messages.assert_called_once()
    call_args = mock_embedding_model.generate_from_messages.call_args[0][0]

    # Should be called with [[assistant_msg1], [assistant_msg2]]
    assert len(call_args) == 2
    assert len(call_args[0]) == 1
    assert len(call_args[1]) == 1
    assert call_args[0][0].content == "Paris is the capital."
    assert call_args[1][0].content == "The capital is Paris."


def test_fuse_messages():
    """Test that messages are fused correctly for iterative generation."""
    task = AidanBench(num_fewshot=0)

    messages = [
        Message(role=Role.USER, content="What is the capital of France?"),
        Message(role=Role.ASSISTANT, content="Paris is the capital."),
        Message(role=Role.ASSISTANT, content="The capital is Paris."),
    ]

    fused = task._fuse_messages(messages)

    assert len(fused) == 1
    assert fused[0].role == Role.USER

    content = fused[0].content
    assert "What is the capital of France?" in content
    assert "HAVE NOT" in content
    assert "previous_answers" in content
    assert "<previous_answer id='1'>" in content
    assert "Paris is the capital." in content
    assert "<previous_answer id='2'>" in content
    assert "The capital is Paris." in content


def test_generate_completions_returns_proper_format():
    """Test that generate_completions returns properly formatted completions."""
    task = AidanBench(num_fewshot=0)

    # Mock the generation loop to return simple message history
    def mock_generation_loop(llm, stop_sequences, max_tokens, initial_samples):
        message_histories = []
        errors = []

        for sample in initial_samples:
            messages = [
                Message(role=Role.USER, content=sample.messages[0].content),
                Message(role=Role.ASSISTANT, content="Response 1"),
                Message(role=Role.ASSISTANT, content="Response 2"),
            ]
            message_histories.append(messages)
            errors.append(None)

        return message_histories, errors

    task._generation_loop = mock_generation_loop

    sample = Sample(
        id=1,
        subject="test",
        ground_truth=None,
        messages=[Message(role=Role.USER, content="What is the capital?")],
        context=None,
        possible_completions=None,
    )

    completions = task.generate_completions(Mock(), [sample])

    assert len(completions) == 1
    completion = completions[0]

    assert completion.id == 1
    assert completion.subject == "test"
    assert completion.ground_truth is None
    assert completion.prompt == "What is the capital?"
    assert len(completion.messages) == 3  # USER + 2 ASSISTANT
    assert completion.completion == "Response 1Response 2"  # Concatenated assistant responses
    assert completion.raw_completion == "Response 1Response 2"
    assert completion.error is None


def test_aidanbench_metric_calculation():
    """Test that the AidanBench metric calculates correctly."""
    metric = AidanBenchMetric()

    # Test with multiple assistant messages
    messages = [
        Message(role=Role.USER, content="Question"),
        Message(role=Role.ASSISTANT, content="Response 1"),
        Message(role=Role.ASSISTANT, content="Response 2"),
        Message(role=Role.ASSISTANT, content="Response 3"),
    ]

    completion = Completion(
        id=1,
        subject="test",
        ground_truth=None,
        prompt="Question",
        prompt_sequence_positions=None,
        messages=messages,
        completion="Final response",
        raw_completion="Final response",
        raw_completion_sequence_positions=None,
        context=None,
        error=None,
    )

    results = metric.calculate(completion)

    assert len(results) == 1
    result = results[0]
    assert result.metric_name == "AidanBench/num_responses"
    # Should be len(messages) - 2 = 4 - 2 = 2 unique responses
    assert result.value == 2
    assert result.higher_is_better is True


def test_aidanbench_metric_with_no_messages():
    """Test metric calculation when messages is None."""
    metric = AidanBenchMetric()

    completion = Completion(
        id=1,
        subject="test",
        ground_truth=None,
        prompt="Question",
        prompt_sequence_positions=None,
        messages=None,
        completion="Final response",
        raw_completion="Final response",
        raw_completion_sequence_positions=None,
        context=None,
        error=None,
    )

    results = metric.calculate(completion)

    assert len(results) == 1
    result = results[0]
    assert result.value == 0


def test_fuse_messages_with_complex_content():
    """Test message fusing with more complex content and XML escaping."""
    task = AidanBench(num_fewshot=0)

    messages = [
        Message(role=Role.USER, content="What are <tags> in HTML?"),
        Message(role=Role.ASSISTANT, content="HTML tags are <element>content</element> structures."),
        Message(role=Role.ASSISTANT, content="They use angle brackets < and > to define elements."),
    ]

    fused = task._fuse_messages(messages)

    assert len(fused) == 1
    assert fused[0].role == Role.USER

    content = fused[0].content
    # Check that original question is preserved
    assert "What are <tags> in HTML?" in content
    # Check that previous answers are included with proper XML structure
    assert "<previous_answer id='1'>" in content
    assert "HTML tags are <element>content</element> structures." in content
    assert "<previous_answer id='2'>" in content
    assert "They use angle brackets < and > to define elements." in content
    assert "</previous_answer>" in content


def test_fuse_messages_preserves_original_instruction():
    """Test that fuse_messages preserves the original instruction without modification."""
    task = AidanBench(num_fewshot=0)

    original_instruction = "Explain quantum mechanics in simple terms."
    messages = [
        Message(role=Role.USER, content=original_instruction),
        Message(role=Role.ASSISTANT, content="Quantum mechanics is about tiny particles."),
    ]

    fused = task._fuse_messages(messages)

    # Original instruction should appear at the beginning of the fused message
    assert fused[0].content.startswith(original_instruction)


@patch("eval_framework.tasks.benchmarks.aidanbench.pairwise_cosine_similarity")
def test_generation_loop_stops_on_low_coherence(mock_cosine_similarity):
    """Test that generation loop stops when coherence is too low."""
    task = AidanBench(num_fewshot=0)

    # Mock the coherence grader to return low score
    mock_grader_result = Mock()
    mock_grader_result.coherence_score = COHERENCE_THRESHOLD - 1  # Below threshold
    task._coherence_grader.grade = Mock(return_value=mock_grader_result)

    # Mock the embedding model (not used when stopping on coherence)
    mock_embedding_model = Mock()
    mock_embedding_model.generate_from_messages.return_value = [[0.1, 0.2, 0.3]]
    task._embedding_model = mock_embedding_model

    # Mock cosine similarity (not used in this case)
    mock_cosine_similarity.return_value = [[0.1]]

    # Mock the parent generate_completions method
    mock_completion = Mock()
    mock_completion.messages = [
        Message(role=Role.USER, content="What is the capital of France?"),
        Message(role=Role.ASSISTANT, content="Incoherent response"),
    ]
    mock_completion.error = None

    with patch.object(task.__class__.__bases__[0], "generate_completions", return_value=[mock_completion]):
        sample = Sample(
            id=1,
            subject="test",
            ground_truth=None,
            messages=[Message(role=Role.USER, content="What is the capital of France?")],
            context=None,
            possible_completions=None,
        )

        message_histories, errors = task._generation_loop(Mock(), None, None, [sample])

    # Should stop after first iteration due to low coherence
    assert len(message_histories) == 1
    assert len(message_histories[0]) == 2  # USER + ASSISTANT
    assert errors[0] is None

    # Verify coherence grader was called
    task._coherence_grader.grade.assert_called_once()


@patch("eval_framework.tasks.benchmarks.aidanbench.pairwise_cosine_similarity")
def test_generation_loop_stops_on_low_novelty(mock_cosine_similarity):
    """Test that generation loop stops when novelty is too low."""
    task = AidanBench(num_fewshot=0)

    # Mock the coherence grader to always return high score
    mock_grader_result = Mock()
    mock_grader_result.coherence_score = COHERENCE_THRESHOLD + 1  # Above threshold
    task._coherence_grader.grade = Mock(return_value=mock_grader_result)

    # Mock the embedding model to return embeddings for novelty calculation
    task._embedding_model.generate_from_messages = Mock(
        return_value=[
            [0.1, 0.2, 0.3],  # First response embedding
            [0.1, 0.21, 0.31],  # Second response embedding (very similar)
        ]
    )

    # Mock cosine similarity to return high similarity (low novelty)
    mock_cosine_similarity.return_value = [[0.95]]  # Very high similarity = low novelty

    # Mock completion for the second iteration
    mock_completion = Mock()
    mock_completion.messages = [
        Message(role=Role.USER, content="Updated question"),
        Message(role=Role.ASSISTANT, content="Very similar response"),
    ]
    mock_completion.error = None

    with patch.object(task.__class__.__bases__[0], "generate_completions", return_value=[mock_completion]):
        sample = Sample(
            id=1,
            subject="test",
            ground_truth=None,
            messages=[Message(role=Role.USER, content="What is the capital?")],
            context=None,
            possible_completions=None,
        )

        message_histories, errors = task._generation_loop(Mock(), None, None, [sample])

    # Should have stopped after second iteration due to low novelty
    assert len(message_histories) == 1
    assert len(message_histories[0]) == 3  # USER + 2 ASSISTANT (stopped after novelty check)
    assert errors[0] is None


@patch("eval_framework.tasks.benchmarks.aidanbench.pairwise_cosine_similarity")
def test_generation_loop_continues_with_high_coherence_and_novelty(mock_cosine_similarity):
    """Test that generation loop continues when both coherence and novelty are high."""
    task = AidanBench(num_fewshot=0)

    # Create a counter to control when to stop (avoid infinite loop)
    call_count = 0

    def mock_coherence_grade(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = Mock()
        if call_count <= 2:  # First 2 iterations: high coherence
            result.coherence_score = COHERENCE_THRESHOLD + 10
        else:  # 3rd iteration: low coherence to stop
            result.coherence_score = COHERENCE_THRESHOLD - 1
        return result

    task._coherence_grader.grade = Mock(side_effect=mock_coherence_grade)

    # Mock embedding model for novelty calculation
    task._embedding_model.generate_from_messages = Mock(
        return_value=[
            [0.1, 0.2, 0.3],  # Previous response embedding
            [0.7, 0.8, 0.9],  # New response embedding (very different)
        ]
    )

    # Mock cosine similarity to return low similarity (high novelty)
    mock_cosine_similarity.return_value = [[0.05]]  # Low similarity = high novelty

    # Mock completion
    mock_completion = Mock()
    mock_completion.messages = [
        Message(role=Role.USER, content="Updated question"),
        Message(role=Role.ASSISTANT, content="Unique response"),
    ]
    mock_completion.error = None

    with patch.object(task.__class__.__bases__[0], "generate_completions", return_value=[mock_completion]):
        sample = Sample(
            id=1,
            subject="test",
            ground_truth=None,
            messages=[Message(role=Role.USER, content="What is the capital?")],
            context=None,
            possible_completions=None,
        )

        message_histories, errors = task._generation_loop(Mock(), None, None, [sample])

    # Should continue for multiple iterations before stopping on low coherence
    assert len(message_histories) == 1
    assert len(message_histories[0]) == 4  # USER + 3 ASSISTANT responses
    assert errors[0] is None

    # Verify multiple coherence checks occurred
    assert task._coherence_grader.grade.call_count == 3


def test_task_constants_and_metadata():
    """Test that the task has correct constants and metadata."""
    task = AidanBench(num_fewshot=0)

    # Test basic task metadata
    assert task.NAME == "AidanBench"
    assert task.DATASET_PATH == "Aleph-Alpha-Research/aidanbench"
    assert task.SAMPLE_SPLIT == "train"
    assert task.FEWSHOT_SPLIT == "train"
    assert task.RESPONSE_TYPE.value == "completion"
    assert AidanBenchMetric in task.METRICS
    assert len(task.SUBJECTS) == 1
    assert task.SUBJECTS[0] == "no_subject"
