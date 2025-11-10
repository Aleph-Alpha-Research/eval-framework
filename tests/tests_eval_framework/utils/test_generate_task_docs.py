import difflib
import filecmp
from pathlib import Path

from eval_framework.utils.generate_task_docs import generate_all_docs, parse_args


def test_task_docs_are_up_to_date(tmp_path: Path) -> None:
    """
    Test that all tasks docs have been generated and are up to date. In particular, checks:
    - the documentation can be generated for all tasks,
    - no documentation still remain from removed tasks,
    - the documentation markdown file contents are up to date.
    """
    # Get the default args
    args = parse_args([])

    generate_all_docs(args=args, output_docs_directory=tmp_path)

    repo_root = Path(__file__).resolve().parents[2]
    repo_docs_path = repo_root / "docs" / "tasks"

    generated = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".md")
    repo_docs = sorted(p.name for p in repo_docs_path.iterdir() if p.suffix == ".md")

    assert generated == repo_docs, f"Generated docs {generated} do not match repo docs {repo_docs}"

    # Check file contents are identical; using shallow=False for full content compare
    diffs = []
    for name in generated:
        gen_file = tmp_path / name
        repo_file = repo_docs_path / name
        if not filecmp.cmp(gen_file, repo_file, shallow=False):
            diffs.append(name)
            with open(gen_file) as gf, open(repo_file) as rf:
                gen_content = gf.read()
                repo_content = rf.read()
                print(f"--- Difference in file: {name} ---")
                print(f"Generated content:\n{gen_content}\n")
                print(f"Repo content:\n{repo_content}\n")

            gen_lines = gen_file.read_text().splitlines(keepends=True)
            repo_lines = repo_file.read_text().splitlines(keepends=True)
            diff = difflib.unified_diff(
                repo_lines, gen_lines, fromfile=f"repo/{name}", tofile=f"generated/{name}", lineterm=""
            )
            print(f"--- Differences for {name} ---")
            for line in diff:
                print(line)
            print()

    assert not diffs, f"Files differ between generated and repo docs:\n{diffs}"
