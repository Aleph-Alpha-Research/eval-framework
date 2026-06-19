import pytest
from datasets.exceptions import DatasetNotFoundError

from eval_framework.tasks.benchmarks.arc import ARC_AllenAI_EN_MC
from eval_framework.tasks.benchmarks.balancedcopa import BalancedCOPA
from eval_framework.tasks.benchmarks.copa import (
    COPA_SuperGLUE_EN_Cloze_EvalHarness,
    COPA_SuperGLUE_EN_Cloze_EvalHarness_IDK,
    COPA_SuperGLUE_EN_MC,
)
from eval_framework.tasks.benchmarks.csqa import (
    CommonsenseQA_Tau_EN_Cloze,
    CommonsenseQA_Tau_EN_MC,
    CommonsenseQA_Tau_EN_MC_OLMES,
    _CommonsenseQA_Tau_EN_Base,
)
from eval_framework.tasks.benchmarks.drop import (
    DROP_AllenAI_EN_Cloze,
    DROP_AllenAI_EN_MC,
    DROP_AllenAI_EN_MC_OLMES,
    DROP_EleutherAI_EN,
)
from eval_framework.tasks.benchmarks.global_mmlu import GlobalMMLU_Cohere_XX_MC
from eval_framework.tasks.benchmarks.goldenswag import GoldenSwag_PleIAs_EN_Cloze, GoldenSwag_PleIAs_EN_Cloze_IDK
from eval_framework.tasks.benchmarks.gpqa import GPQA_Idavidrein_EN_MC_OLMES
from eval_framework.tasks.benchmarks.humaneval import HumanEval_OpenAI_EN_BPB
from eval_framework.tasks.benchmarks.lab_bench import LabBenchCloze, LabBenchMC, LabBenchMC_OLMES
from eval_framework.tasks.benchmarks.math_reasoning import (
    HendrycksMath_EleutherAI_EN_BPB,
    HendrycksMath_EleutherAI_EN_EvalHarness,
    HendrycksMath_EleutherAI_EN_EvalHarnessRelaxed,
    MATH500_HuggingFaceH4_EN_OLMES,
)
from eval_framework.tasks.benchmarks.mbpp import MBPP_Google_EN_BPB
from eval_framework.tasks.benchmarks.medqa import (
    MedQA_DavidHeineman_EN_Cloze,
    MedQA_DavidHeineman_EN_MC,
    MedQA_DavidHeineman_EN_MC_OLMES,
)
from eval_framework.tasks.benchmarks.mmlu import MMLU_CAIS_EN_MC_OLMES
from eval_framework.tasks.benchmarks.mmlu_pro import MMLUPro_TIGERLab_EN_MC_OLMES
from eval_framework.tasks.benchmarks.naturalqs_open import (
    NaturalQsOpen_AllenAI_EN_Cloze,
    NaturalQsOpen_AllenAI_EN_MC,
    NaturalQsOpen_AllenAI_EN_MC_OLMES,
    NaturalQsOpen_Google_EN,
)
from eval_framework.tasks.benchmarks.openbookqa import (
    OPENBOOKQA_EVAL_HARNESS_OLMES,
    OPENBOOKQA_OLMES,
)
from eval_framework.tasks.benchmarks.piqa import PIQA_YBisk_EN_MC
from eval_framework.tasks.benchmarks.sciq import (
    SciQ_AllenAI_EN_Cloze_IDK,
    SciQ_AllenAI_EN_Cloze_IDK_EvalHarness,
    SciQ_AllenAI_EN_MC,
)
from eval_framework.tasks.benchmarks.social_iqa import (
    SocialIQa_AllenAI_EN_Cloze,
    SocialIQa_AllenAI_EN_MC,
    SocialIQa_AllenAI_EN_MC_OLMES,
)
from eval_framework.tasks.benchmarks.squad import SQuAD2_Stanford_EN_BPB
from eval_framework.tasks.benchmarks.truthfulqa import TRUTHFULQA_OLMES
from eval_framework.tasks.benchmarks.winogrande import WinoGrande_AllenAI_EN_MC


def _smoke_test_task(task_cls) -> None:
    task = task_cls()
    samples = list(task.iterate_samples(num_samples=2))
    assert len(samples) > 0
    for sample in samples:
        assert sample.id is not None
        assert isinstance(sample.subject, str)
        assert sample.messages


@pytest.mark.cpu_slow
@pytest.mark.slow_download
def test_csqa_tasks_smoke() -> None:
    _smoke_test_task(_CommonsenseQA_Tau_EN_Base)
    _smoke_test_task(CommonsenseQA_Tau_EN_Cloze)
    _smoke_test_task(CommonsenseQA_Tau_EN_MC)
    _smoke_test_task(CommonsenseQA_Tau_EN_MC_OLMES)


@pytest.mark.cpu_slow
def test_drop_tasks_smoke() -> None:
    _smoke_test_task(DROP_EleutherAI_EN)
    _smoke_test_task(DROP_AllenAI_EN_MC)
    _smoke_test_task(DROP_AllenAI_EN_MC_OLMES)
    _smoke_test_task(DROP_AllenAI_EN_Cloze)


@pytest.mark.cpu_slow
@pytest.mark.slow_download
def test_lab_bench_tasks_smoke() -> None:
    _smoke_test_task(LabBenchCloze)
    _smoke_test_task(LabBenchMC)
    _smoke_test_task(LabBenchMC_OLMES)


@pytest.mark.cpu_slow
def test_naturalqs_open_tasks_smoke() -> None:
    _smoke_test_task(NaturalQsOpen_Google_EN)
    _smoke_test_task(NaturalQsOpen_AllenAI_EN_Cloze)
    _smoke_test_task(NaturalQsOpen_AllenAI_EN_MC)
    _smoke_test_task(NaturalQsOpen_AllenAI_EN_MC_OLMES)


@pytest.mark.cpu_slow
@pytest.mark.slow_download
def test_math_minerva_tasks_smoke() -> None:
    _smoke_test_task(HendrycksMath_EleutherAI_EN_EvalHarness)
    _smoke_test_task(HendrycksMath_EleutherAI_EN_EvalHarnessRelaxed)
    _smoke_test_task(HendrycksMath_EleutherAI_EN_BPB)
    _smoke_test_task(MATH500_HuggingFaceH4_EN_OLMES)


@pytest.mark.cpu_slow
def test_social_iqa_tasks_smoke() -> None:
    try:
        _smoke_test_task(SocialIQa_AllenAI_EN_Cloze)
        _smoke_test_task(SocialIQa_AllenAI_EN_MC)
        _smoke_test_task(SocialIQa_AllenAI_EN_MC_OLMES)
    except RuntimeError as e:
        if "no longer supported" in str(e) or "loading script" in str(e).lower():
            pytest.skip("allenai/social_i_qa uses a dataset loading script not supported by this datasets version")
        raise


@pytest.mark.cpu_slow
def test_medqa_tasks_smoke() -> None:
    _smoke_test_task(MedQA_DavidHeineman_EN_Cloze)
    _smoke_test_task(MedQA_DavidHeineman_EN_MC)
    _smoke_test_task(MedQA_DavidHeineman_EN_MC_OLMES)


@pytest.mark.cpu_slow
@pytest.mark.slow_download
def test_olmes_variants_smoke() -> None:
    for task_cls in (
        ARC_AllenAI_EN_MC,
        COPA_SuperGLUE_EN_MC,
        GPQA_Idavidrein_EN_MC_OLMES,  # gated; skipped when not authenticated
        MMLU_CAIS_EN_MC_OLMES,
        MMLUPro_TIGERLab_EN_MC_OLMES,
        OPENBOOKQA_OLMES,
        OPENBOOKQA_EVAL_HARNESS_OLMES,
        PIQA_YBisk_EN_MC,
        TRUTHFULQA_OLMES,
        WinoGrande_AllenAI_EN_MC,
    ):
        try:
            _smoke_test_task(task_cls)
        except DatasetNotFoundError as e:
            if "gated" in str(e).lower():
                continue  # skip this task only when gated and not authenticated
            raise


@pytest.mark.cpu_slow
def test_balanced_copa_and_copa_harness_smoke() -> None:
    _smoke_test_task(BalancedCOPA)
    _smoke_test_task(COPA_SuperGLUE_EN_Cloze_EvalHarness)
    _smoke_test_task(COPA_SuperGLUE_EN_Cloze_EvalHarness_IDK)


@pytest.mark.cpu_slow
def test_goldenswag_tasks_smoke() -> None:
    _smoke_test_task(GoldenSwag_PleIAs_EN_Cloze)
    _smoke_test_task(GoldenSwag_PleIAs_EN_Cloze_IDK)


@pytest.mark.cpu_slow
@pytest.mark.slow_download
def test_global_mmlu_smoke() -> None:
    _smoke_test_task(GlobalMMLU_Cohere_XX_MC)


@pytest.mark.cpu_slow
def test_humaneval_bpb_smoke() -> None:
    _smoke_test_task(HumanEval_OpenAI_EN_BPB)


@pytest.mark.cpu_slow
def test_mbpp_bpb_smoke() -> None:
    _smoke_test_task(MBPP_Google_EN_BPB)


@pytest.mark.cpu_slow
def test_sciq_olmes_tasks_smoke() -> None:
    _smoke_test_task(SciQ_AllenAI_EN_MC)
    _smoke_test_task(SciQ_AllenAI_EN_Cloze_IDK)
    _smoke_test_task(SciQ_AllenAI_EN_Cloze_IDK_EvalHarness)


@pytest.mark.cpu_slow
def test_squad2_bpb_smoke() -> None:
    _smoke_test_task(SQuAD2_Stanford_EN_BPB)
