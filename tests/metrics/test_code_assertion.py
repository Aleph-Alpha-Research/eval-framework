import pytest

from eval_framework.metrics.completion.code_assertion import CodeCompletionAssertion
from eval_framework.shared.types import Completion, Error


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                raw_completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    return (A*x) // B
try:
    assert floor_Min(10,20,30) == 15
    assert floor_Min(1,2,1) == 0
    assert floor_Min(11,10,9) == 9
    score = True
except:
    score = False
print(score)
""",
                raw_completion_sequence_positions=None,
                completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    return (A*x) // B
try:
    assert floor_Min(10,20,30) == 15
    assert floor_Min(1,2,1) == 0
    assert floor_Min(11,10,9) == 9
    score = True
except:
    score = False
print(score)
""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            1.0,
            id="code_execution_ok",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                raw_completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    return (A*x) // B
try:
    assert floor_Min(10,20,30) == 16
    assert floor_Min(1,2,1) == -1
    assert floor_Min(11,10,9) == 8
    score = True
except:
    score = False
print(score)
""",
                raw_completion_sequence_positions=None,
                completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    return (A*x) // B
try:
    assert floor_Min(10,20,30) == 16
    assert floor_Min(1,2,1) == -1
    assert floor_Min(11,10,9) == 8
    score = True
except:
    score = False
print(score)
""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="code_execution_error",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                raw_completion="""
try:
    assert floor_Min(10,20,30) == 16
    assert floor_Min(1,2,1) == -1
    assert floor_Min(11,10,9) == 8
    score = True
except:
    score = False
print(score)
""",
                raw_completion_sequence_positions=None,
                completion="""
try:
    assert floor_Min(10,20,30) == 16
    assert floor_Min(1,2,1) == -1
    assert floor_Min(11,10,9) == 8
    score = True
except:
    score = False
print(score)
""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="code_execution_error_missing_function",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                raw_completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    x = x +
try:
    assert floor_Min(10,20,30) == 16
    assert floor_Min(1,2,1) == -1
    assert floor_Min(11,10,9) == 8
    score = True
except:
    score = False
print(score)
""",
                raw_completion_sequence_positions=None,
                completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    x = x +
try:
    assert floor_Min(10,20,30) == 16
    assert floor_Min(1,2,1) == -1
    assert floor_Min(11,10,9) == 8
    score = True
except:
    score = False
print(score)
""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="code_execution_error_syntax",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                raw_completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    x = x +
""",
                raw_completion_sequence_positions=None,
                completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    x = x +
""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="code_execution_error_incomplete",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                raw_completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    return (A*x) // B £ 1
try:
    assert floor_Min(10,20,30) == 15
    assert floor_Min(1,2,1) == 0
    assert floor_Min(11,10,9) == 9
    score = True
except:
    score = False
print(score)
""",
                raw_completion_sequence_positions=None,
                completion="""def floor_Min(A,B,N):
    x = max(B - 1,N)
    return (A*x) // B £ 1
try:
    assert floor_Min(10,20,30) == 15
    assert floor_Min(1,2,1) == 0
    assert floor_Min(11,10,9) == 9
    score = True
except:
    score = False
print(score)
""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="code_execution_error_illegal_character",
        ),
        # 1. response.error is not None
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                raw_completion="",
                raw_completion_sequence_positions=None,
                completion="",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                error=Error(error_class="TestError", message="Something went wrong", traceback="traceback details"),
            ),
            None,
            id="response_has_error",
        ),
        # 2. unrelated print
        pytest.param(
            Completion(
                id=2,
                subject="test",
                ground_truth="",
                raw_completion="""print("done")""",
                raw_completion_sequence_positions=None,
                completion="""print("done")""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="unrelated_print",
        ),
        # 3. multiple prints, last True
        pytest.param(
            Completion(
                id=3,
                subject="test",
                ground_truth="",
                raw_completion="""print("one")\nprint("two")\nprint(True)""",
                raw_completion_sequence_positions=None,
                completion="""print("one")\nprint("two")\nprint(True)""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            1.0,
            id="multiple_prints_last_true",
        ),
        # 4. multiple prints, last False
        pytest.param(
            Completion(
                id=4,
                subject="test",
                ground_truth="",
                raw_completion="""print("check")\nprint(False)""",
                raw_completion_sequence_positions=None,
                completion="""print("check")\nprint(False)""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="multiple_prints_last_false",
        ),
        # 5. no print at all
        pytest.param(
            Completion(
                id=5,
                subject="test",
                ground_truth="",
                raw_completion="""def foo(): return 1+1""",
                raw_completion_sequence_positions=None,
                completion="""def foo(): return 1+1""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="no_print",
        ),
        # 6. syntax error before print
        pytest.param(
            Completion(
                id=6,
                subject="test",
                ground_truth="",
                raw_completion="""def foo()""",
                raw_completion_sequence_positions=None,
                completion="""def foo()""",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
            ),
            0.0,
            id="syntax_error",
        ),
    ],
)
def test_code_assertion(response: Completion, expected_value: float) -> None:
    metric = CodeCompletionAssertion()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].metric_name == "Code Completion Accuracy"
    assert results[0].higher_is_better is True
    if expected_value is not None:
        assert results[0].code_execution_trace is not None
        assert results[0].value == pytest.approx(expected_value)
    else:
        # means we expected an error
        assert results[0].value is None
        assert results[0].error is not None
