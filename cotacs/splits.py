"""
Create consistent data splits for CoTACS experiments.

Splits the FDT dataset into train/dev/eval sets by problem ID.
The same splits are used across all decision theories (CDT, FDT, EDT)
for apples-to-apples comparison.

Usage:
    python -m cotacs.splits --output data/cotacs/splits.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str | Path) -> list[dict]:
    """Load the FDT dataset from JSON."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data["problems"]


def save_json(path: Path, data: dict) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_splits(
    dataset_path: str | Path,
    train_problems: int = 11,
    dev_problems: int = 22,
    seed: int = 42,
) -> dict[str, list[int]]:
    """
    Create train/dev/eval splits by problem ID.

    Args:
        dataset_path: Path to fdt.json
        train_problems: Number of problems for training (program generation)
        dev_problems: Number of problems for dev (program ranking)
        seed: Random seed for reproducibility

    Returns:
        Dict with train_ids, dev_ids, eval_ids lists
    """
    problems = load_dataset(dataset_path)
    problem_ids = [p["id"] for p in problems]

    logger.info(f"Loaded {len(problem_ids)} problems from {dataset_path}")

    # Shuffle with fixed seed
    rng = random.Random(seed)
    shuffled_ids = problem_ids.copy()
    rng.shuffle(shuffled_ids)

    # Split
    train_ids = sorted(shuffled_ids[:train_problems])
    dev_ids = sorted(shuffled_ids[train_problems:train_problems + dev_problems])
    eval_ids = sorted(shuffled_ids[train_problems + dev_problems:])

    logger.info(f"Split: train={len(train_ids)}, dev={len(dev_ids)}, eval={len(eval_ids)}")
    logger.info(f"Train problem IDs: {train_ids}")
    logger.info(f"Dev problem IDs: {dev_ids}")

    return {
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "eval_ids": eval_ids,
        "config": {
            "train_problems": train_problems,
            "dev_problems": dev_problems,
            "seed": seed,
            "total_problems": len(problem_ids),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create consistent data splits for CoTACS experiments"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/fdt.json",
        help="Path to fdt.json dataset"
    )
    parser.add_argument(
        "--output", type=str, default="data/cotacs/splits.json",
        help="Output path for splits JSON"
    )
    parser.add_argument(
        "--train-problems", type=int, default=33,
        help="Number of problems for training (default: 33 = 99 schemas)"
    )
    parser.add_argument(
        "--dev-problems", type=int, default=66,
        help="Number of problems for dev (default: 66 = 198 schemas)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    splits = create_splits(
        dataset_path=args.dataset,
        train_problems=args.train_problems,
        dev_problems=args.dev_problems,
        seed=args.seed,
    )

    output_path = Path(args.output)
    save_json(output_path, splits)
    logger.info(f"Saved splits to {output_path}")


if __name__ == "__main__":
    main()
