"""
Merge worker output files from parallel FDT evaluation into a single results file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def merge_results(
    input_dir: str | Path = "data/tmp",
    output_path: str | Path = "data/results.json",
    expected_count: int = 999,
    cleanup: bool = False,
) -> dict:
    """
    Merge worker result files into a single JSON.

    Args:
        input_dir: Directory containing results_*.json files
        output_path: Path for merged output
        expected_count: Expected number of problems (for validation)
        cleanup: Whether to delete worker files after successful merge

    Returns:
        Merged results dict
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    result_files = sorted(input_dir.glob("results_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No results_*.json files found in {input_dir}")

    logger.info(f"Found {len(result_files)} worker result files")

    merged: dict = {}
    for result_file in result_files:
        with open(result_file, "r") as f:
            worker_results = json.load(f)

        # Check for duplicate keys
        overlap = set(merged.keys()) & set(worker_results.keys())
        if overlap:
            logger.warning(f"Duplicate problem IDs found: {overlap}")

        merged.update(worker_results)
        logger.info(f"Loaded {len(worker_results)} problems from {result_file.name}")

    logger.info(f"Total merged: {len(merged)} problems")

    # Validate completeness
    if len(merged) < expected_count:
        missing = expected_count - len(merged)
        logger.warning(f"Missing {missing} problems. Expected {expected_count}, got {len(merged)}")

        # Find which IDs are missing
        all_ids = set(str(i) for i in range(expected_count))
        present_ids = set(merged.keys())
        missing_ids = sorted(all_ids - present_ids, key=int)
        if len(missing_ids) <= 20:
            logger.warning(f"Missing problem IDs: {missing_ids}")
        else:
            logger.warning(f"Missing problem IDs (first 20): {missing_ids[:20]}...")
    elif len(merged) > expected_count:
        logger.warning(f"More problems than expected: {len(merged)} > {expected_count}")
    else:
        logger.info("All problems present")

    # Count ambiguous results
    ambiguous_count = 0
    for problem_id, schemas in merged.items():
        for schema_key, result in schemas.items():
            if result["choice"] == -1:
                ambiguous_count += 1

    if ambiguous_count > 0:
        logger.warning(f"Found {ambiguous_count} ambiguous results (choice=-1)")

    # Save merged results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info(f"Merged results saved to {output_path}")

    # Cleanup worker files
    if cleanup:
        for result_file in result_files:
            result_file.unlink()
            logger.info(f"Deleted {result_file.name}")

        # Also clean up any leftover checkpoints
        for checkpoint in input_dir.glob("checkpoint_*.json"):
            checkpoint.unlink()
            logger.info(f"Deleted {checkpoint.name}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge FDT evaluation results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/tmp",
        help="Directory containing results_*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results.json",
        help="Output path for merged results",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=999,
        help="Expected number of problems",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete worker files after successful merge",
    )

    args = parser.parse_args()

    merge_results(
        input_dir=args.input_dir,
        output_path=args.output,
        expected_count=args.expected_count,
        cleanup=args.cleanup,
    )


if __name__ == "__main__":
    main()
