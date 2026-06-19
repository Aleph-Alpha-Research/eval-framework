import logging
import random
import re
import time
from enum import Enum
from typing import NamedTuple

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import register_lazy_task, registered_tasks_iter

logger = logging.getLogger(__name__)


class TaskNameEnum(Enum):
    @property
    def value(self) -> type[BaseTask]:
        return super().value


# Task name grammar: {Dataset}_{Source}_{Language}[_{Style}][_{Variant}][_{Subset}] (docs/task_naming.md).
_TOKEN = r"[A-Z][A-Za-z0-9]*"
_LANGUAGE = r"[A-Z]{2}"
_NAMING_RE = re.compile(
    rf"^(?P<dataset>{_TOKEN})_(?P<source>{_TOKEN})_(?P<language>{_LANGUAGE})(?P<rest>(?:_{_TOKEN})*)$"
)
KNOWN_STYLES = ("MC", "Cloze", "BPB", "PartialEval")  # Closed vocabulary for Style.


class ParsedTaskName(NamedTuple):
    """The parts of a task name. ``style`` is set only when a recognized style token is present."""

    dataset: str
    source: str
    language: str
    style: str | None
    variants: tuple[str, ...]


def parse_task_name(name: str) -> ParsedTaskName | None:
    """Split a task name into its parts, or return ``None`` if it doesn't follow the convention."""
    match = _NAMING_RE.match(name)
    if match is None:
        return None
    rest = [token for token in match.group("rest").split("_") if token]
    style = rest[0] if rest and rest[0] in KNOWN_STYLES else None
    variants = tuple(rest[1:]) if style is not None else tuple(rest)
    return ParsedTaskName(match.group("dataset"), match.group("source"), match.group("language"), style, variants)


def register_all_tasks() -> None:
    """Register all the benchmark tasks with the eval framework."""
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2024_HuggingFaceH4_EN")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2025_MathAI_EN")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2026_MathAI_EN")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_AllenAI_EN_Cloze")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_AllenAI_EN_Cloze_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_AllenAI_EN_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.arc_de.ARC_LeoLM_DE_Cloze")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBench_BigCode_EN_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA_SuperGLUE_EN_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GoldenSwag_PleIAs_EN_Cloze")
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GoldenSwag_PleIAs_EN_Cloze_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_Idavidrein_EN_MC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8K_OpenAI_EN_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.GSM8K_OpenAI_EN_Reasoning")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HellaSwag_Rowan_EN_Cloze")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HellaSwag_Rowan_EN_Cloze_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval_OpenAI_EN_BPB")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval_OpenAI_EN_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEval_Google_EN")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEval_JZhang_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH500_HuggingFaceH4_EN")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.HendrycksMath_EleutherAI_EN_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_HumanEval_Cpp")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_HumanEval_Java")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_HumanEval_JS")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_HumanEval_PHP")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_HumanEval_Rust")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_HumanEval_Bash")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_MBPP_Cpp")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_MBPP_Java")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_MBPP_JS")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_MBPP_PHP")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_MBPP_Rust")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLE_NUPRL_EN_MBPP_Bash")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_Google_EN_BPB")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_Google_EN_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_CAIS_EN_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_CAIS_EN_MC_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_CAIS_EN_MC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLUPro_TIGERLab_EN_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLUPro_TIGERLab_EN_MC_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLUPro_TIGERLab_EN_MC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLUPro_TIGERLab_EN_COTMC")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_CAIS_EN_COTMC")
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU_Cohere_XX_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU_Cohere_DE_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_YBisk_EN_Cloze")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_YBisk_EN_Cloze_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_YBisk_EN_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.sciq.SciQ_AllenAI_EN_MC")
    register_lazy_task("eval_framework.tasks.benchmarks.social_iqa.SocialIQa_AllenAI_EN_MC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD_Stanford_EN_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.triviaqa.TriviaQA_MandarJoshi_EN")
    register_lazy_task("eval_framework.tasks.benchmarks.winogrande.WinoGrande_AllenAI_EN_PartialEval")
    register_lazy_task("eval_framework.tasks.benchmarks.csqa.CommonsenseQA_Tau_EN_MC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DROP_EleutherAI_EN_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DROP_AllenAI_EN_MC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpen_Google_EN")
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpen_AllenAI_EN_MC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.medqa.MedQA_DavidHeineman_EN_MC_OLMES")
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
