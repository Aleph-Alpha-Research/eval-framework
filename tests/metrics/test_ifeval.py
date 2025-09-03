from eval_framework.metrics.completion.ifeval import IFEvalMetric, IFEvalMetricContext
from eval_framework.shared.types import Completion


def test_ifeval_metric() -> None:
    metric = IFEvalMetric()

    context = IFEvalMetricContext(
        key=1000,
        instruction_id_list=[
            "punctuation:no_comma",
            "detectable_format:number_highlighted_sections",
            "length_constraints:number_words",
        ],
        prompt="Write a 300 word summary ....",
        additional_kwargs=[{}, {"num_highlights": 2}, {"relation": "at least", "num_words": 300}],
    )

    completion = Completion(
        id=1,
        subject="test",
        ground_truth="some ground truth",
        prompt=context.prompt,
        prompt_sequence_positions=None,
        messages=None,
        completion="*highlighted section part 1*\nLorem.\n\n*highlighted section part 2*\nIpsum.",
        raw_completion="*highlighted section part 1*\nLorem.\n\n*highlighted section part 2*\nIpsum.",
        raw_completion_sequence_positions=None,
        context=context,
    )

    results = metric.calculate(completion)
    assert len(results) == 8

    inst_level_strict_acc = 0.0
    inst_level_loose_acc = 0.0

    for result in results:
        assert result.value is not None
        match result.metric_name:
            case "IFEval/prompt_level_strict_acc":
                assert result.value == 0.0
            case "IFEval/prompt_level_loose_acc":
                assert result.value == 0.0
            case "IFEval/inst_level_strict_acc":
                inst_level_strict_acc += result.value
            case "IFEval/inst_level_loose_acc":
                inst_level_loose_acc += result.value

    assert inst_level_strict_acc == 2.0  # punctuation and number_highlighted_sections succeeded
    assert inst_level_loose_acc == 2.0
