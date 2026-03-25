import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval_framework.suite import (
    SuiteAggregate,
    SuiteResult,
    TaskSuite,
    compute_aggregates,
    resolve_to_evalconfig_kwargs,
    run_suite,
)


@pytest.fixture(autouse=True)
def _skip_registry_check():
    # This is a hack to skip the check for whether the task is registered.
    with patch("eval_framework.suite.is_registered", return_value=True):
        yield


class TestTaskSuiteValidation:
    def test_leaf_suite(self) -> None:
        s = TaskSuite(name="my-mmlu", tasks="MMLU", num_fewshot=5)
        assert s.is_leaf
        assert s.name == "my-mmlu"
        assert s.num_fewshot == 5

        s = TaskSuite(tasks="MMLU")
        assert s.is_leaf
        assert s.name == "MMLU"
        assert s.task_name == "MMLU"

    def test_composite_suite(self) -> None:
        s = TaskSuite(
            name="mytasks",
            tasks=[
                TaskSuite(tasks="GSM8K"),
                TaskSuite(tasks="Math"),
            ],
        )
        assert not s.is_leaf
        assert s.name == "mytasks"
        assert len(s.tasks) == 2
        assert s == TaskSuite(name="mytasks", tasks=["GSM8K", "Math"])

    def test_mixed_bare_strings_and_suites(self) -> None:
        s = TaskSuite(
            name="mixed",
            tasks=[
                "MMLU",
                TaskSuite(tasks="GSM8K", max_tokens=512),
            ],
        )
        assert len(s.tasks) == 2
        assert s.tasks[0].task_name == "MMLU"
        assert s.tasks[1].max_tokens == 512

    def test_empty_tasks_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            TaskSuite(name="empty", tasks=[])

    def test_composite_without_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must have a 'name'"):
            TaskSuite(tasks=[TaskSuite(tasks="MMLU")])

    def test_nested_suites(self) -> None:
        s = TaskSuite(
            name="top",
            tasks=[
                TaskSuite(
                    name="sub",
                    tasks=[TaskSuite(tasks="A"), TaskSuite(tasks="B")],
                ),
                TaskSuite(tasks="C"),
            ],
        )
        assert not s.is_leaf
        assert not s.tasks[0].is_leaf
        assert s.tasks[1].is_leaf
        assert s.tasks[0].tasks[0].is_leaf
        assert s.tasks[0].tasks[1].is_leaf


def test_load_nested(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        name: top
        temperature: 0.0
        tasks:
            - tasks: MMLU
            - name: taskgroup
              tasks:
                - tasks: GSM8K
                - tasks: Math
              aggregates:
                - name: taskgroup_score
                  metric: Average Accuracy
                  method: mean
    """)
    f = tmp_path / "suite.yaml"
    f.write_text(yaml_content)
    s = TaskSuite.load_from_yaml(f)
    assert s.name == "top"
    assert len(s.tasks) == 2
    sub = s.tasks[1]
    assert isinstance(sub, TaskSuite)
    assert sub.name == "taskgroup"
    assert len(sub.tasks) == 2


class TestLoadFromPy:
    def test_load_from_py(self, tmp_path: Path) -> None:
        py_content = textwrap.dedent("""\
            from eval_framework.suite import TaskSuite, SuiteAggregate

            suite = TaskSuite(
                name="test-suite",
                tasks=[
                    TaskSuite(tasks="MMLU", num_fewshot=5),
                    TaskSuite(tasks="GSM8K", max_tokens=512),
                ],
                aggregates=[
                    SuiteAggregate(name="overall", metric="Average Accuracy"),
                ],
            )
        """)
        f = tmp_path / "my_suite.py"
        f.write_text(py_content)
        s = TaskSuite.load_from_py(f)
        assert s.name == "test-suite"
        assert len(s.tasks) == 2
        assert len(s.aggregates) == 1

    def test_load_missing_suite_variable(self, tmp_path: Path) -> None:
        py_content = "x = 42\n"
        f = tmp_path / "bad_suite.py"
        f.write_text(py_content)
        with pytest.raises(ValueError, match="must define a 'suite' variable"):
            TaskSuite.load_from_py(f)

    def test_load_wrong_type(self, tmp_path: Path) -> None:
        py_content = "suite = 'not a TaskSuite'\n"
        f = tmp_path / "wrong_type.py"
        f.write_text(py_content)
        with pytest.raises(TypeError, match="must be a TaskSuite instance"):
            TaskSuite.load_from_py(f)


class TestResolveToEvalKwargs:
    def test_routes_temperature_to_llm_args(self) -> None:
        leaf = TaskSuite(tasks="MMLU")
        defaults = {"temperature": 0.7, "top_p": 0.9, "num_fewshot": 5}
        cli_kwargs = {
            "llm_name": "MyModel",
            "llm_args": {"apikey": "apivalue"},
            "models": "models.py",
            "task_name": "ignored",
        }
        result = resolve_to_evalconfig_kwargs(leaf, defaults, cli_kwargs)

        assert result["task_name"] == "MMLU"
        assert result["num_fewshot"] == 5
        assert result["llm_args"]["temperature"] == 0.7
        assert result["llm_args"]["top_p"] == 0.9
        assert result["llm_args"]["apikey"] == "apivalue"

    def test_extra_llm_args_merged(self) -> None:
        leaf = TaskSuite(tasks="T")
        defaults = {"extra_llm_args": {"custom_param": 42}}
        cli_kwargs = {"llm_args": {}, "task_name": "old"}
        result = resolve_to_evalconfig_kwargs(leaf, defaults, cli_kwargs)
        assert result["llm_args"]["custom_param"] == 42

    def test_cli_kwargs_preserved(self) -> None:
        leaf = TaskSuite(tasks="T")
        defaults = {}
        cli_kwargs = {
            "llm_name": "M",
            "llm_args": {},
            "task_name": "old",
            "wandb_project": "proj",
        }
        result = resolve_to_evalconfig_kwargs(leaf, defaults, cli_kwargs)
        assert result["llm_name"] == "M"
        assert result["wandb_project"] == "proj"
        assert result["task_name"] == "T"


class TestComputeAggregates:
    def _result(self, name: str, **aggregates: float | None) -> SuiteResult:
        return SuiteResult(name=name, aggregates=aggregates)

    def test_mean(self) -> None:
        aggs = [SuiteAggregate(name="score", metric="Average Accuracy", method="mean")]
        children = {
            "A": self._result("A", **{"Average Accuracy": 0.8}),
            "B": self._result("B", **{"Average Accuracy": 0.6}),
        }
        result = compute_aggregates(aggs, children)
        assert result["score"] == pytest.approx(0.7)

    def test_median(self) -> None:
        aggs = [SuiteAggregate(name="score", metric="m", method="median")]
        children = {
            "A": self._result("A", m=1.0),
            "B": self._result("B", m=3.0),
            "C": self._result("C", m=2.0),
        }
        result = compute_aggregates(aggs, children)
        assert result["score"] == pytest.approx(2.0)

    def test_missing_metric_returns_none(self) -> None:
        aggs = [SuiteAggregate(name="score", metric="NonExistent")]
        children = {"A": self._result("A", Other=1.0)}
        result = compute_aggregates(aggs, children)
        assert result["score"] is None
        assert "Other" not in result

    def test_unlisted_metrics_not_in_result(self) -> None:
        aggs = [SuiteAggregate(name="avg", metric="m", method="mean")]
        children = {
            "A": self._result("A", m=0.5, extra=2.0),
            "B": self._result("B", m=0.7, extra=2.0),
        }
        result = compute_aggregates(aggs, children)
        assert result["avg"] == pytest.approx(0.6)
        assert "extra" not in result

    def test_explicit_passthrough(self) -> None:
        aggs = [
            SuiteAggregate(name="avg", metric="m", method="mean"),
            SuiteAggregate(name="A", metric="extra", method="passthrough"),
        ]
        children = {
            "A": self._result("A", m=0.5, extra=2.0),
            "B": self._result("B", m=0.7),
        }
        result = compute_aggregates(aggs, children)
        assert result["avg"] == pytest.approx(0.6)
        assert result["A"] == pytest.approx(2.0)

    def test_passthrough_metric_omitted_on_conflict(self) -> None:
        aggs = [SuiteAggregate(name="avg", metric="m", method="mean")]
        children = {
            "A": self._result("A", m=0.5, extra=1.0),
            "B": self._result("B", m=0.7, extra=2.0),
        }
        result = compute_aggregates(aggs, children)
        assert result["avg"] == pytest.approx(0.6)
        assert "extra" not in result

    def test_none_metric_value_skipped(self) -> None:
        aggs = [SuiteAggregate(name="score", metric="m", method="mean")]
        children = {
            "A": self._result("A", m=0.8),
            "B": self._result("B", m=None),
        }
        result = compute_aggregates(aggs, children)
        assert result["score"] == pytest.approx(0.8)

    def test_aggregate_with_nested_suite_results(self) -> None:
        aggs = [SuiteAggregate(name="overall", metric="sub_score", method="mean")]
        children = {
            "task_A": self._result("task_A", sub_score=0.9),
            "sub_suite": SuiteResult(
                name="sub",
                task_results={"inner": self._result("inner", sub_score=0.5)},
                aggregates={"sub_score": 0.5},
            ),
        }
        result = compute_aggregates(aggs, children)
        assert result["overall"] == pytest.approx(0.7)

    def test_multiple_aggregates(self) -> None:
        aggs = [
            SuiteAggregate(name="score", metric="Average Accuracy", method="mean"),
            SuiteAggregate(name="score2", metric="Average Accuracy", method="mean"),
        ]
        children = {
            "A": self._result("A", **{"Average Accuracy": 0.8}),
            "B": self._result("B", **{"Average Accuracy": 0.6}),
        }
        result = compute_aggregates(aggs, children)
        assert result["score"] == result["score2"] == pytest.approx(0.7)


_BASE_CLI_KWARGS: dict = {
    "llm_name": "MockLLM",
    "llm_args": {},
    "models": "models.py",
    "task_name": None,
    "output_dir": "outputs",
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_run_id": None,
    "wandb_upload_results": True,
    "wandb_group": None,
}


class TestRunSuite:
    """Tests for run_suite() with _run_single_task and _load_aggregated_results patched."""

    def test_flat_suite_calls_each_leaf(self, tmp_path: Path) -> None:
        suite = TaskSuite(
            name="list_suite",
            tasks=[
                TaskSuite(tasks="TaskA"),
                TaskSuite(tasks="TaskB"),
                TaskSuite(tasks="TaskC"),
            ],
        )

        mock_run = MagicMock()
        mock_load = MagicMock(return_value={"Average Accuracy": 0.123})

        cli_kwargs = {**_BASE_CLI_KWARGS, "output_dir": str(tmp_path)}

        with (
            patch("eval_framework.suite._run_single_task", mock_run),
            patch("eval_framework.suite._load_aggregated_results", mock_load),
        ):
            result = run_suite(suite, cli_kwargs)

        assert mock_run.call_count == 3
        assert mock_load.call_count == 3

        called_task_names = [c.args[0]["task_name"] for c in mock_run.call_args_list]
        assert called_task_names == ["TaskA", "TaskB", "TaskC"]

        assert result.name == "list_suite"
        assert len(result.task_results) == 3
        assert all(r.aggregates["Average Accuracy"] == 0.123 for r in result.task_results.values())

    def test_nested_suite_recurses_depth_first(self, tmp_path: Path) -> None:
        suite = TaskSuite(
            name="top",
            tasks=[
                TaskSuite(tasks="Solo"),
                TaskSuite(
                    name="group",
                    tasks=[
                        TaskSuite(tasks="Inner1"),
                        TaskSuite(tasks="Inner2"),
                    ],
                    aggregates=[
                        SuiteAggregate(name="group_score", metric="m", method="mean"),
                    ],
                ),
            ],
            aggregates=[
                SuiteAggregate(name="overall", metric="m", method="mean"),
            ],
        )

        call_order: list[str] = []

        def run_to_collect_task_names(kwargs: dict) -> None:
            call_order.append(kwargs["task_name"])

        mock_load = MagicMock(return_value={"m": 0.5})

        cli_kwargs = {**_BASE_CLI_KWARGS, "output_dir": str(tmp_path)}

        with (
            patch("eval_framework.suite._run_single_task", side_effect=run_to_collect_task_names),
            patch("eval_framework.suite._load_aggregated_results", mock_load),
        ):
            result = run_suite(suite, cli_kwargs)

        assert call_order == ["Solo", "Inner1", "Inner2"]

        assert result.name == "top"
        assert "Solo" in result.task_results
        assert "group" in result.task_results

        group_result = result.task_results["group"]
        assert group_result.aggregates["group_score"] == pytest.approx(0.5)

        assert result.aggregates["overall"] == pytest.approx(0.5)
