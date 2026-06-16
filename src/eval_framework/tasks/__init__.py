# Register all tasks on import
from pathlib import Path

from .benchmarks.dataset_revisions import DatasetRevision
from .task_names import register_all_tasks

DatasetRevision.add_revision_file(Path(__file__).parent / "benchmarks" / "task-dataset-revisions.json")

register_all_tasks()

del register_all_tasks
del DatasetRevision
