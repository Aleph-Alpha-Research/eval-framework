from pathlib import Path

import pandas as pd
import pytest

from eval_framework.main import main
from eval_framework.task_names import TaskName
from eval_framework.tasks.eval_config import EvalConfig
from tests.conftest import MockLLM


@pytest.mark.parametrize("save_intermediate_results", [True, False])
def test_preemption(tmp_path: Path, save_intermediate_results: bool) -> None:
    mock_llm = MockLLM()
    output_dir = tmp_path / "no_preempt"
    output_dir_preempt = tmp_path / "preemption"
    preemption_counter = 0

    def preemption() -> bool:
        nonlocal preemption_counter
        preemption_counter += 1
        if preemption_counter == 15:
            return True
        return False

    eval_preempt_config = EvalConfig(
        task_name=TaskName.ARC,
        num_fewshot=0,
        num_samples=10,
        output_dir=output_dir_preempt,
        llm_class=mock_llm.__class__,
    )
    eval_config = EvalConfig(
        task_name=TaskName.ARC,
        num_fewshot=0,
        num_samples=10,
        output_dir=output_dir,
        llm_class=mock_llm.__class__,
        save_intermediate_results=save_intermediate_results,
    )

    fake_trial_id = 42
    main(mock_llm, eval_preempt_config, preemption, fake_trial_id)
    assert mock_llm.logprob_counter == 15
    main(mock_llm, eval_preempt_config, preemption, fake_trial_id)
    assert mock_llm.logprob_counter == 20

    main(mock_llm, eval_config)
    preempt_results = pd.read_json(list(output_dir_preempt.rglob("results.jsonl"))[0], lines=True)
    no_preempt_results = pd.read_json(list(output_dir.rglob("results.jsonl"))[0], lines=True)
    assert all(preempt_results == no_preempt_results)
