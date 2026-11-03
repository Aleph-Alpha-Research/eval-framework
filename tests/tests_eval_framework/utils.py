import hashlib
import json
import logging
from pathlib import Path

from eval_framework.result_processors.base import Result
from eval_framework.utils.constants import RED, RESET

HASHES_FILE = Path(__file__).parent / "benchmarks" / "task-prompts-hashes.json"

logger = logging.getLogger(__name__)


def pretty_print(results: list[Result]) -> None:
    for result in results:
        logger.info(f"{RED}{result.llm_name} |{result.metric_name}: {result.value} | {result.subject}{RESET}")


def _almost_equal(x: float, y: float) -> bool:
    return 2 * abs(x - y) / abs(x + y + 1e-5) < 1e-4


def assert_hash_string(task_name: str, suffix_key: str, tested_string: str) -> str:
    """
    Compute the MD5 hash of a formatted sample or another string and verify or store it in a JSON file.
    """
    data_to_hash = tested_string.encode() if isinstance(tested_string, str) else tested_string
    tested_string_hash = hashlib.md5(data_to_hash).hexdigest()

    key = f"{task_name}.{suffix_key}"

    # Safely load the JSON dictionary of hashes
    if HASHES_FILE.exists():
        with HASHES_FILE.open("r", encoding="utf-8") as f:
            all_hashes = json.load(f)
    else:
        all_hashes = {}

    if key in all_hashes:
        expected_hash = all_hashes[key]

        if expected_hash != tested_string_hash:
            print("\n" + "=" * 80)
            print(f"HASH MISMATCH for: {key}")
            print("=" * 80)
            print(f"\nExpected hash: {expected_hash}")
            print(f"Actual hash:   {tested_string_hash}")
            print("\n--- ACTUAL OUTPUT ---")
            print(tested_string)
            print("--- END ACTUAL OUTPUT ---")
            print("\nNote: Expected output is not stored (only hash).")
            print("=" * 80 + "\n")

            assert False, (
                f"Hash mismatch for key: {key}\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {tested_string_hash}\n"
                f"See actual output above."
            )
    else:
        print("\n" + "=" * 80)
        print(f"NEW HASH ENTRY: {key}")
        print("=" * 80)
        print(f"Hash: {tested_string_hash}")
        print("\n--- STRING BEING HASHED ---")
        print(tested_string)
        print("--- END ---")
        print("=" * 80 + "\n")

        all_hashes[key] = tested_string_hash

        with HASHES_FILE.open("w", encoding="utf-8") as f:
            json.dump(dict(sorted(all_hashes.items())), f, indent=4, ensure_ascii=False)

        assert False, f"Hash for key '{key}' not found in {HASHES_FILE}. It was added for future runs."

    return tested_string_hash
