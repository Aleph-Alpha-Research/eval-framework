"""Constituent BPB metrics, split out of ``BitsPerByteVariantsLoglikelihood``.

Each piece of the old variants facade is its own ``BaseMetric`` so a task can list exactly the BPB
flavours it wants through the ``metrics()`` contract, rather than getting all of them bundled. Taken
together with ``BitsPerByteLoglikelihood`` (the ``BitsPerByte`` headline) these emit the same metric
names and values as the variants class.

- ``CorpusBitsPerByteLoglikelihood`` -> the ``BitsPerByte_{bits,bytes,tokens,nAliases}`` companion
  fields that feed corpus BPB aggregation.
- ``PrefixBitsPerByteLoglikelihood`` -> the four ``PrefixBPB*`` results.
- ``PriorBitsPerByteLoglikelihood`` -> no per-response results; it exists so a task can declare that
  it wants Prior BPB, which is aggregated at the corpus level.
"""

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.loglikelihood.bpb_variants_common import (
    compute_corpus_bpb_sidecar_results,
    compute_prefix_bpb_results_for_response,
)
from eval_framework.shared.types import Loglikelihood


class CorpusBitsPerByteLoglikelihood(BaseMetric[Loglikelihood]):
    """Per-item inputs to corpus BPB: ``BitsPerByte_{bits,bytes,tokens,nAliases}``.

    The ``BitsPerByte`` headline itself is emitted by ``BitsPerByteLoglikelihood``, not here.

    The corpus-level roll-up of these fields (``Corpus BPB`` etc.) is currently computed by
    ``EvaluationGenerator`` rather than by this metric; see the collection-level TODO there.
    """

    NAME = "CorpusBitsPerByte"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        return compute_corpus_bpb_sidecar_results(response)


class PrefixBitsPerByteLoglikelihood(BaseMetric[Loglikelihood]):
    """Prefix BPB: ``PrefixBPB``, ``PrefixBPB_rate``, ``PrefixBPB_entryCost``, ``PrefixBPB_contentBytes``.

    Emits nothing when no ground truth is found or the backend reported no per-token logprobs, which
    is what the variants class contributed for prefix in those cases.
    """

    NAME = "PrefixBitsPerByte"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        return compute_prefix_bpb_results_for_response(response)


class PriorBitsPerByteLoglikelihood(BaseMetric[Loglikelihood]):
    """Prior BPB marker metric.

    Prior BPB is a corpus-level quantity with no per-response value, so ``calculate`` returns no
    results. Listing this metric on a task declares that the run should emit Prior BPB.

    The Prior BPB aggregation is currently computed by ``EvaluationGenerator`` rather than by this
    metric; see the collection-level TODO there.
    """

    NAME = "PriorBitsPerByte"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        return []
