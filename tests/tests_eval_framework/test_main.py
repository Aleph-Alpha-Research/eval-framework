from pathlib import Path

from eval_framework.main import Phase, main
from eval_framework.result_processors.result_processor import generate_output_dir
from eval_framework.tasks.eval_config import EvalConfig
from tests.tests_eval_framework.conftest import MockLLM


def _config(output_dir: Path) -> EvalConfig:
    return EvalConfig(task_name="ARC", num_samples=3, output_dir=output_dir, llm_class=MockLLM)


def test_generate_phase_stops_before_scoring(tmp_path: Path) -> None:
    llm, config = MockLLM(), _config(tmp_path)

    results = main(llm, config, phase=Phase.GENERATE)

    output_dir = generate_output_dir(llm.name, config)
    assert results == []
    assert (output_dir / "output.jsonl").exists()
    assert not (output_dir / "results.jsonl").exists()
    assert not (output_dir / "aggregated_results.json").exists()
