import logging
import random
import time
from enum import Enum

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import register_lazy_task, registered_tasks_iter

logger = logging.getLogger(__name__)


class TaskNameEnum(Enum):
    @property
    def value(self) -> type[BaseTask]:
        return super().value


def register_all_tasks() -> None:
    """Register all the benchmark tasks with the eval framework."""
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2024")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2025")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.arc_de.ARC_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.ARC_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.ARC_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.arc_fi.ARC_FI")
    register_lazy_task("eval_framework.tasks.benchmarks.balancedcopa.BalancedCOPA")
    register_lazy_task("eval_framework.tasks.benchmarks.belebele.BELEBELE")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBench")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchInstruct")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchHard")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchHardInstruct")
    register_lazy_task("eval_framework.tasks.benchmarks.casehold.CASEHOLD")
    register_lazy_task("eval_framework.tasks.benchmarks.chembench.ChemBench")
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPAEvalHarness")
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA")
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA_IDKEvalHarness")
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.duc.DUC_ABSTRACTIVE")
    register_lazy_task("eval_framework.tasks.benchmarks.duc.DUC_EXTRACTIVE")
    register_lazy_task("eval_framework.tasks.benchmarks.flores200.Flores200")
    register_lazy_task("eval_framework.tasks.benchmarks.flores_plus.FloresPlus", extras=["comet"])
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA")
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8K")
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8KEvalHarness")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.GSM8KReasoning")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.GSM8K_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.GSM8K_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag_de.HELLASWAG_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.HELLASWAG_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.HELLASWAG_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEvalInstruct")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEval")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEvalDe")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEvalFiSv")
    register_lazy_task("eval_framework.tasks.benchmarks.include.INCLUDE")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_CodeDebug")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_CodeRun")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnDia")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnMC")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnQA")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_MathFind")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrieveKV2")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrieveNumber")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrievePassKey1")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHLvl5")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH500")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_SANITIZED")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_PROMPT_WITHOUT_TESTS")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_PROMPT_WITHOUT_TESTS_SANITIZED")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.FullTextMMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.MMLU_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.MMLU_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_de.MMLU_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.mmmlu.MMMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.mmmlu.MMMLU_GERMAN_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.pawsx.PAWSX")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.openbookqa.OPENBOOKQA")
    register_lazy_task("eval_framework.tasks.benchmarks.openbookqa.OPENBOOKQA_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.openbookqa.OPENBOOKQA_EVAL_HARNESS")
    register_lazy_task("eval_framework.tasks.benchmarks.sciq.SCIQ")
    register_lazy_task("eval_framework.tasks.benchmarks.sciq.SCIQEvalHarness")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQUAD")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQUAD2")
    register_lazy_task("eval_framework.tasks.benchmarks.tablebench.TableBench")
    register_lazy_task("eval_framework.tasks.benchmarks.triviaqa.TRIVIAQA")
    register_lazy_task("eval_framework.tasks.benchmarks.truthfulqa.TRUTHFULQA")
    register_lazy_task("eval_framework.tasks.benchmarks.truthfulqa.TRUTHFULQA_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.TRUTHFULQA_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.TRUTHFULQA_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.winogender.WINOGENDER")
    register_lazy_task("eval_framework.tasks.benchmarks.winogender.WINOGENDER_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.winogrande.WINOGRANDE")
    register_lazy_task("eval_framework.tasks.benchmarks.winogrande.WINOGRANDE_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.winox.WINOX_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.winox.WINOX_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT14")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT16")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT20")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT14_INSTRUCT")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT16_INSTRUCT")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT20_INSTRUCT")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QUALITY")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_SQUALITY")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QMSUM")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QASPER")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_GOV_REPORT")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_NARRATIVEQA")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_MUSIQUE")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_SPACE_DIGEST")
    register_lazy_task("eval_framework.tasks.benchmarks.quality.QUALITY")
    register_lazy_task("eval_framework.tasks.benchmarks.sphyr.SPHYR")
    register_lazy_task("eval_framework.tasks.benchmarks.struct_eval.StructEval")
    register_lazy_task("eval_framework.tasks.benchmarks.struct_eval.RenderableStructEval")
    register_lazy_task("eval_framework.tasks.benchmarks.aidanbench.AidanBench", extras=["openai"])
    register_lazy_task("eval_framework.tasks.benchmarks.aidanbench.AidanBenchOriginal", extras=["openai"])

    try:
        # Importing the companion registers the additional tasks from the module.
        # This is mostly for convenience for internal use-cases
        import eval_framework_companion  # noqa
    except ImportError:
        pass


def get_datasets_needing_update() -> tuple[bool, set[str]]:
    """
    Check which HuggingFace datasets need updating by comparing
    current HF Hub commits with cached commits in dataset_commits.json.

    Returns:
        Tuple of (all_up_to_date, set_of_dataset_paths_needing_update)
    """
    import json
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    cache_dir = Path(os.environ.get("HF_DATASET_CACHE_DIR", str(Path.home() / ".cache" / "huggingface" / "datasets")))
    cache_file = cache_dir / "dataset_commits.json"

    api = HfApi()
    current_commits: dict[str, str] = {}
    datasets_needing_update: set[str] = set()

    print("Checking HuggingFace dataset versions...")
    for task_name, task_class in registered_tasks_iter():
        dataset_path = getattr(task_class, "DATASET_PATH", None)
        if dataset_path and dataset_path not in current_commits:
            try:
                info = api.dataset_info(dataset_path)
                assert info.sha is not None, f"No SHA for {dataset_path}"
                current_commits[dataset_path] = info.sha
                print(f"  {dataset_path}: {info.sha[:8]}")
            except Exception:
                print(f"  {dataset_path}: SKIPPED (not on HF Hub, uses external source)")
                continue  # Don't add to current_commits at all

    # No cache file = need to download everything
    if not cache_file.exists():
        print("\nNo cached commits found - full download needed")
        return False, set(current_commits.keys())

    with open(cache_file) as f:
        cached_commits = json.load(f)

    # Compare each dataset's current commit with cached commit
    for dataset_path, current_sha in current_commits.items():
        cached_sha = cached_commits.get(dataset_path)
        if cached_sha != current_sha:
            cached_short = cached_sha[:8] if cached_sha else "NEW"
            current_short = current_sha[:8] if current_sha != "error" else "ERROR"
            print(f"  UPDATE NEEDED: {dataset_path}: {cached_short} -> {current_short}")
            datasets_needing_update.add(dataset_path)

    if datasets_needing_update:
        print(f"\n{len(datasets_needing_update)} dataset(s) need updating")
        return False, datasets_needing_update

    print("\nAll datasets are up to date!")
    return True, set()


def make_sure_all_hf_datasets_are_in_cache(only_datasets: set[str] | None = None) -> None:
    """
    Download datasets to cache.

    Args:
        only_datasets: If provided, only process tasks using these dataset paths.
                       If None, process all tasks.
    """
    for task_name, task_class in registered_tasks_iter():
        dataset_path = getattr(task_class, "DATASET_PATH", None)

        # Skip if filtering is enabled and this dataset isn't in the update list
        if only_datasets is not None and dataset_path not in only_datasets:
            logger.info(f"Skipping {task_name} - dataset {dataset_path} is up to date")
            continue

        task = task_class()
        for attempt in range(10):
            try:
                for _ in task.iterate_samples(num_samples=1):
                    pass
                break
            except Exception as e:
                logger.info(f"{e} Will retry loading {task_name} in a few seconds, attempt #{attempt + 1}.")
                time.sleep(random.randint(1, 5))
        logger.info(f"Processed {task_name}")

    # Sacrebleu uses its own cache (SACREBLEU env var), separate from HF datasets.
    # We cache them together to ensure all evaluation data is available.
    _ensure_sacrebleu_datasets_cached()


def update_changed_datasets_only(verbose: bool = True) -> tuple[bool, set[str]]:
    """
    Check for updates and download only changed datasets.

    Args:
        verbose: If True, print detailed summary of updated datasets.

    Returns:
        Tuple of (updates_were_made, set_of_updated_dataset_paths).
    """
    all_up_to_date, datasets_to_update = get_datasets_needing_update()

    if all_up_to_date:
        # Even when HF datasets are current, ensure sacrebleu is cached
        # (it has its own cache and isn't tracked by dataset_commits.json)
        _ensure_sacrebleu_datasets_cached()
        print("Nothing to update!")
        return False, set()

    print(f"\nDownloading {len(datasets_to_update)} updated dataset(s)...")
    make_sure_all_hf_datasets_are_in_cache(only_datasets=datasets_to_update)

    if verbose:
        print("\n" + "=" * 60)
        print("DATASETS UPDATED:")
        print("=" * 60)
        for dataset_path in sorted(datasets_to_update):
            print(f"  ✓ {dataset_path}")
        print("=" * 60)
        print(f"Total: {len(datasets_to_update)} dataset(s) updated")
        print("=" * 60 + "\n")

    return True, datasets_to_update


def save_hf_dataset_commits() -> None:
    """Save current HuggingFace dataset commits after download."""
    import json
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    api = HfApi()
    commits = {}

    print("Saving dataset commit hashes...")
    for task_name, task_class in registered_tasks_iter():
        dataset_path = getattr(task_class, "DATASET_PATH", None)

        if dataset_path and dataset_path not in commits:
            try:
                info = api.dataset_info(dataset_path)
                commits[dataset_path] = info.sha
            except Exception:
                pass

    cache_dir = Path(os.environ.get("HF_DATASET_CACHE_DIR", str(Path.home() / ".cache" / "huggingface" / "datasets")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "dataset_commits.json"

    with open(cache_file, "w") as f:
        json.dump(commits, f, indent=2)

    print(f"Saved {len(commits)} dataset commits to {cache_file}")


def _ensure_sacrebleu_datasets_cached() -> None:
    """Pre-download sacrebleu WMT datasets to ensure they're cached.

    Sacrebleu uses its own cache (controlled by SACREBLEU env var).
    This ensures WMT test sets are downloaded and cached alongside HF datasets.
    """
    import sacrebleu

    # WMT datasets used by the framework (from wmt.py)
    WMT_DATASETS = {
        "wmt14": ["en-fr", "fr-en"],
        "wmt16": ["de-en", "en-de"],
        "wmt20": ["de-en", "de-fr", "en-de", "fr-de"],
    }

    print("Ensuring sacrebleu WMT datasets are cached...")
    for test_set, langpairs in WMT_DATASETS.items():
        for langpair in langpairs:
            try:
                sacrebleu.download_test_set(test_set=test_set, langpair=langpair)
                print(f"  {test_set}/{langpair}: OK")
            except Exception as e:
                print(f"  {test_set}/{langpair}: FAILED ({e})")

    print("Sacrebleu datasets cached!")


if __name__ == "__main__":
    print(list(registered_tasks_iter()))
