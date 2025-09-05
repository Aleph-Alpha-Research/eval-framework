#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "seaborn>=0.13.2,<0.14",
# ]
# ///
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ----------

# Default plot title
TITLE = "Evaluation results"

# Exclude the following models from the plot
EXCLUDE_MODELS: list[str] = []

# Exclude the following tasks from the plot
EXCLUDE_TASKS: list[str] = []

# Aliases for model names (makes the plot labels more readable)
CHECKPOINT_ALIASES: dict[str, str] = {}

# Aliases for task names (makes the plot labels more readable)
TASK_ALIASES: dict[str, str] = {}

# Aliases for the metric names (makes the plot labels more readable)
METRIC_ALIASES: dict[str, str] = {
    "Average Accuracy Loglikelihood": "Accuracy Loglikelihood",
    "Average Accuracy Completion": "Accuracy Completion",
    "Average pairwise_judgement": "Pairwise Score",
}

# Pick from this list the first metric present in the benchmark results
PREFERED_METRICS: list[str] = [
    # "Average Bytes per Sequence Position",  # uncomment to plot bytes per sequence position (compression ratio)
    "Average Accuracy Loglikelihood",
    "Average Accuracy Completion",
    "Average pairwise_judgement",
]

# ----------


def filter_results(
    results: list, exclude_models: list[str] = [], exclude_tasks: list[str] = [], only_tasks: list[str] = []
) -> list:
    filtered_results = []
    for res in results:
        if res["checkpoint"] in exclude_models or res["checkpoint"] in EXCLUDE_MODELS:
            continue
        if res["task"] in exclude_tasks or res["task"] in EXCLUDE_TASKS:
            continue
        if only_tasks and res["task"] not in only_tasks:
            continue
        filtered_results.append(res)
    return filtered_results


def parse_evals(folder: Path, plot_metric: str) -> list:
    """Return list of parsed results dicts from folder."""

    results = []
    seen_keys = set()
    print(f"Parsing evaluation results in: {folder}")
    for json_path in folder.glob("**/aggregated_results.json"):
        if True:
            checkpoint, benchdir, fewshotdir = json_path.relative_to(folder).parts[:3]
            task = re.sub(r"^v0\.1\.\d+_", "", benchdir)
            fewshot = int(re.findall(r"fewshot_(\d+)", fewshotdir)[0])

            # Load metadata and check it match the fewshot and task extracted from the path
            with open(json_path.parent / "metadata.json") as mf:
                metadata = json.load(mf)
            assert metadata["num_fewshot"] == fewshot, (
                f"Metadata mismatch for {json_path}: expected fewshot {fewshot}, got {metadata['num_fewshot']}"
            )
            assert metadata["task_name"] == task, (
                f"Metadata mismatch for {json_path}: expected {metadata['task_name']}, got {task}"
            )

            with open(json_path) as f:
                data = json.load(f)

            # Decide which metric to use
            if plot_metric:
                if plot_metric in data:
                    metric_name = plot_metric
                else:
                    print(f"Warning: Specified metric '{plot_metric}' not found in {json_path}. Skipping.")
                    continue

            else:
                metric_name = next((m for m in PREFERED_METRICS if m in data), "no metric found")
                if not metric_name:
                    print(f"Warning: No matching metric found in {json_path}")
                    continue

            if metric_name == "no metric found":
                print(f"No valid metric found in {json_path}. Skipping.")

            else:
                # Check for duplicate results
                key = (checkpoint, task, fewshot)
                if key in seen_keys:
                    print(f"Warning: Duplicate results for {key}")
                    continue
                seen_keys.add(key)

                results.append(
                    {
                        "checkpoint": checkpoint,
                        "task": task,
                        "fewshot": fewshot,
                        "metric": metric_name,
                        "value": float(data.get(metric_name)),
                        "stderr": float(data.get(f"StdErr {metric_name}", 0)),
                        "err_free_ratio": float(data.get(f"ErrorFreeRatio {metric_name}", 0)),
                        "json": str(json_path),
                    }
                )

    return results


def replace_names_by_aliases(results: list) -> list:
    """Replace model and task names in results with aliases if available."""

    models = {res["checkpoint"] for res in results}
    tasks = {res["task"] for res in results}

    print("Models found:")
    for model in models:
        print(f"  {model} -> {CHECKPOINT_ALIASES[model]}" if model in CHECKPOINT_ALIASES else f"  {model}")

    print("Tasks found:")
    for task in tasks:
        print(f"  {task} -> {TASK_ALIASES[task]}" if task in TASK_ALIASES else f"  {task}")

    for res in results:
        res["checkpoint"] = CHECKPOINT_ALIASES.get(res["checkpoint"], res["checkpoint"])
        res["task"] = TASK_ALIASES.get(res["task"], res["task"])
        res["metric"] = METRIC_ALIASES.get(res["metric"], res["metric"])

    # Sort results by checkpoint and task in the order they appear in the alias listed (or alphabetically otherwise)
    results.sort(
        key=lambda x: (
            CHECKPOINT_ALIASES.get(x["checkpoint"], x["checkpoint"]),
            TASK_ALIASES.get(x["task"], x["task"]),
            x["fewshot"],
            METRIC_ALIASES.get(x["metric"], x["metric"]),
        )
    )

    return results


def make_barplot(results: list, outfile: str, add_stderr: bool) -> None:
    if not results:
        print("No data to plot.")
        return

    # Group by (task, fewshot, metric)
    groupdict = defaultdict(list)
    for r in results:
        key = (r["task"], r["fewshot"], r["metric"])
        groupdict[key].append(r)

    n_groups = len(groupdict)
    if n_groups == 0:
        print("Nothing to plot after grouping.")
        return

    ncols = min(5, n_groups)
    nrows = math.ceil(n_groups / ncols)
    figsize = (ncols * 7, nrows * 5)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Turn off unused axes
    for ax in axes[n_groups:]:
        ax.axis("off")

    for idx, ((task, fewshot, metric), reslist) in enumerate(sorted(groupdict.items())):
        checkpoints = [r["checkpoint"] for r in reslist]
        values = [r["value"] for r in reslist]
        stderrs = [r["stderr"] if r["stderr"] is not None else 0 for r in reslist]
        # Sort alphabetically by checkpoint name
        sorted_chks, sorted_vals, sorted_stds = zip(*sorted(zip(checkpoints, values, stderrs), reverse=True))
        palette = sns.color_palette("tab10", n_colors=len(sorted_chks))
        # Plot with error bars if needed
        axes[idx].bar(
            sorted_chks,
            sorted_vals,
            color=palette,
            yerr=sorted_stds if add_stderr else None,
            capsize=5 if add_stderr else 0,
        )
        axes[idx].set_title(f"{task} ({fewshot}-shot)", fontsize=12)
        axes[idx].set_ylabel(metric)
        axes[idx].set_xticklabels(sorted_chks, rotation=45, ha="right", fontsize=9)

        # Set ylim between min and max values
        min_val = max(min(sorted_vals) * 0.9, 0)
        max_val = min(max(sorted_vals) * 1.1, 1)
        axes[idx].set_ylim(min_val, max_val)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(TITLE, fontsize=20, weight="bold", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    plt.savefig(outfile, dpi=150)
    print(f"Saved plot: {outfile}")
    plt.close(fig)


def write_task_rows_markdown_table(results: list, outfile: str = "table.md") -> None:
    """Generate a markdown table with a row for each task and columns for models (plus fewshot and metric)."""

    if not results:
        print("No data to write as markdown table.")
        return

    # Gather all unique model names, sorted for stable order
    models = sorted({r["checkpoint"] for r in results})

    # Gather unique (task, fewshot, metric) triples
    task_rows = sorted({(r["task"], r["fewshot"], r["metric"]) for r in results})

    # Build lookup: (task, fewshot, metric, model) -> value
    lookup = {}
    for r in results:
        key = (r["task"], r["fewshot"], r["metric"], r["checkpoint"])
        lookup[key] = r["value"]

    # Compose markdown table header
    header = ["Task", "Fewshot", "Metric"] + models
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    # For each (task, fewshot, metric), print values for each model
    for task, fewshot, metric in task_rows:
        line = [task, str(fewshot), metric]
        for model in models:
            value = lookup.get((task, fewshot, metric, model))
            cell = f"{value:.4f}" if value is not None else ""
            line.append(cell)
        lines.append("| " + " | ".join(line) + " |")

    # Write to file
    table_md = "\n".join(lines)
    with open(outfile, "w") as f:
        f.write(table_md)
    print(f"Saved markdown table: {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot task evaluation results as one multi-subplot image.")
    parser.add_argument("--folder", type=Path, required=True, help="Root directory containing results")
    parser.add_argument("--output", type=str, required=False, default="plot.png", help="Output plot file (eg. PNG)")
    parser.add_argument("--add-stderr", action="store_true", help="Include StdErr as error bars in the plot.")
    parser.add_argument("--plot-metric", type=str, default=None, help="Specify the metric used for plotting.")
    parser.add_argument("--exclude-models", nargs="*", default=[], help="List of checkpoints to exclude from the plot.")
    parser.add_argument("--exclude-tasks", nargs="*", default=[], help="List of tasks to exclude from the plot.")
    parser.add_argument("--only-tasks", nargs="*", default=[], help="List of tasks to include in the plot only.")
    parser.add_argument(
        "--output-table", type=str, required=False, default="table.md", help="Output table file (Markdown)"
    )

    args = parser.parse_args()
    results = parse_evals(args.folder, args.plot_metric)
    results = filter_results(
        results=results,
        exclude_models=list(args.exclude_models),
        exclude_tasks=list(args.exclude_tasks),
        only_tasks=list(args.only_tasks),
    )
    results = replace_names_by_aliases(results)
    make_barplot(results, args.output, add_stderr=args.add_stderr)
    write_task_rows_markdown_table(results, outfile=args.output_table)


if __name__ == "__main__":
    main()
