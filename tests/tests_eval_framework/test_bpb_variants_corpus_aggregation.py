from eval_framework.evaluation_generator import EvaluationGenerator
from eval_framework.result_processors.base import Result
from eval_framework.shared.types import Error


def _sidecar_result(
    *,
    id_: int,
    metric_name: str,
    value: float,
    subject: str = "subj",
) -> Result:
    return Result(
        id=id_,
        subject=subject,
        num_fewshot=0,
        llm_name="test",
        task_name="task",
        metric_class_name="BitsPerByteLoglikelihood",
        metric_name=metric_name,
        key=None,
        value=value,
        higher_is_better=False,
        error=None,
    )


def test_corpus_aggregation_from_sidecars():
    results = [
        _sidecar_result(id_=1, metric_name="BitsPerByte_bits", value=10.0),
        _sidecar_result(id_=1, metric_name="BitsPerByte_bytes", value=5.0),
        _sidecar_result(id_=2, metric_name="BitsPerByte_bits", value=6.0),
        _sidecar_result(id_=2, metric_name="BitsPerByte_bytes", value=3.0),
    ]
    aggregated = EvaluationGenerator._aggregate_corpus_bpb_metrics(results)
    assert aggregated["Corpus BPB"] == (10.0 + 6.0) / (5.0 + 3.0)
    assert "Corpus BPB (mean-of-ratios)" not in aggregated


def test_corpus_aggregation_skips_errored_sidecars():
    bad = _sidecar_result(id_=3, metric_name="BitsPerByte_bits", value=99.0)
    bad.error = Error(error_class="ValueError", message="bad", traceback="")
    results = [
        _sidecar_result(id_=1, metric_name="BitsPerByte_bits", value=8.0),
        _sidecar_result(id_=1, metric_name="BitsPerByte_bytes", value=4.0),
        bad,
    ]
    aggregated = EvaluationGenerator._aggregate_corpus_bpb_metrics(results)
    assert aggregated["Corpus BPB"] == 2.0
