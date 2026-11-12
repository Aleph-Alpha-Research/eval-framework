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


def test_score_phase_scores_saved_responses_without_the_model(tmp_path: Path) -> None:
    config = _config(tmp_path / "split")
    main(MockLLM(), config, phase=Phase.GENERATE)
    idle_llm = MockLLM()

    results = main(idle_llm, config, phase=Phase.SCORE)

    assert results
    assert idle_llm.logprob_counter == 0
    assert idle_llm.generate_counter == 0


def test_phases_run_apart_score_like_a_single_run(tmp_path: Path) -> None:
    split_config, single_config = _config(tmp_path / "split"), _config(tmp_path / "single")
    main(MockLLM(), split_config, phase=Phase.GENERATE)
    main(MockLLM(), split_config, phase=Phase.SCORE)

    main(MockLLM(), single_config)

    split_dir = generate_output_dir(MockLLM().name, split_config)
    single_dir = generate_output_dir(MockLLM().name, single_config)
    for filename in ("results.jsonl", "aggregated_results.json"):
        assert (split_dir / filename).read_text() == (single_dir / filename).read_text()
