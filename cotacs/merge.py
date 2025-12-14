"""
CoTACS merge results - Final step.

Merges worker results, computes accuracy vs ground truth for the theory,
and outputs final statistics.

Usage:
    python -m cotacs.merge \
        --theory FDT \
        --input-dir data/cotacs/FDT \
        --output data/cotacs/FDT/final_results.json \
        --programs data/cotacs/FDT/programs.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    """Load data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="CoTACS merge results"
    )
    parser.add_argument(
        "--theory", type=str, required=True, choices=["CDT", "FDT", "EDT"],
        help="Decision theory"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing results_*.json files"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: {input-dir}/final_results.json)"
    )
    parser.add_argument(
        "--programs", type=str, default=None,
        help="Path to programs.json to include in output"
    )
    parser.add_argument(
        "--expected-count", type=int, default=None,
        help="Expected number of problems (for validation)"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Remove worker files after merging"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output) if args.output else input_dir / "final_results.json"

    # Load all results
    results_files = sorted(input_dir.glob("results_*.json"))
    if not results_files:
        logger.error(f"No results_*.json files found in {input_dir}")
        return

    logger.info(f"Found {len(results_files)} result files")

    # Merge results
    merged = {}
    duplicates = []
    for results_file in results_files:
        try:
            results = load_json(results_file)
            for problem_id, schemas in results.items():
                if problem_id in merged:
                    duplicates.append(problem_id)
                merged[problem_id] = schemas
        except Exception as e:
            logger.warning(f"Failed to load {results_file}: {e}")

    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate problem IDs: {duplicates[:10]}...")

    logger.info(f"Merged {len(merged)} problems")

    # Validate count
    if args.expected_count and len(merged) != args.expected_count:
        logger.warning(
            f"Expected {args.expected_count} problems, got {len(merged)}. "
            f"Missing: {args.expected_count - len(merged)}"
        )

    # Compute statistics
    total_schemas = 0
    correct = 0
    ambiguous = 0
    schema_stats = {"A": {"correct": 0, "total": 0}, "B": {"correct": 0, "total": 0}, "C": {"correct": 0, "total": 0}}

    for problem_id, schemas in merged.items():
        for schema_key, result in schemas.items():
            total_schemas += 1
            schema_stats[schema_key]["total"] += 1

            choice = result["choice"]
            ground_truth = result["ground_truth"]

            if choice == -1:
                ambiguous += 1
            elif choice == ground_truth:
                correct += 1
                schema_stats[schema_key]["correct"] += 1

    accuracy = correct / total_schemas if total_schemas > 0 else 0.0

    logger.info(f"Results for {args.theory}:")
    logger.info(f"  Total schemas: {total_schemas}")
    logger.info(f"  Correct: {correct}")
    logger.info(f"  Ambiguous: {ambiguous}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  By schema:")
    for schema_key in ["A", "B", "C"]:
        stats = schema_stats[schema_key]
        schema_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        logger.info(f"    {schema_key}: {stats['correct']}/{stats['total']} ({schema_acc:.4f})")

    # Load programs if provided
    programs_data = None
    if args.programs:
        programs_path = Path(args.programs)
        if programs_path.exists():
            programs_data = load_json(programs_path)

    # Prepare output
    output_data = {
        "theory": args.theory,
        "statistics": {
            "total_problems": len(merged),
            "total_schemas": total_schemas,
            "correct": correct,
            "ambiguous": ambiguous,
            "accuracy": accuracy,
            "by_schema": {
                schema_key: {
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                }
                for schema_key, stats in schema_stats.items()
            }
        },
        "results": merged,
    }

    if programs_data:
        output_data["programs"] = programs_data["programs"]

    save_json(output_path, output_data)
    logger.info(f"Saved final results to {output_path}")

    # Cleanup
    if args.cleanup:
        for results_file in results_files:
            results_file.unlink()
            logger.info(f"Removed {results_file}")


if __name__ == "__main__":
    main()
