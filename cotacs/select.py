"""
CoTACS program selection - Phase 2.

Merges search results from all workers, filters by alpha threshold,
ranks by dev accuracy, and selects top-k programs.

Usage:
    python -m cotacs.select \
        --theory FDT \
        --input-dir data/cotacs/FDT \
        --output data/cotacs/FDT/programs.json \
        --alpha 0.5 \
        --k 1
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
        description="CoTACS program selection (Phase 2)"
    )
    parser.add_argument(
        "--theory", type=str, required=True, choices=["CDT", "FDT", "EDT"],
        help="Decision theory"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing search_*.json files"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for programs.json (default: {input-dir}/programs.json)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Training accuracy threshold for filtering"
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="Number of top programs to select"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output) if args.output else input_dir / "programs.json"

    # Load all search results
    search_files = sorted(input_dir.glob("search_*.json"))
    if not search_files:
        logger.error(f"No search_*.json files found in {input_dir}")
        return

    logger.info(f"Found {len(search_files)} search result files")

    programs = []
    for search_file in search_files:
        try:
            result = load_json(search_file)
            programs.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {search_file}: {e}")

    logger.info(f"Loaded {len(programs)} programs")

    # Log statistics before filtering
    train_accs = [p["train_accuracy"] for p in programs]
    dev_accs = [p["dev_accuracy"] for p in programs]
    logger.info(f"Train accuracy: min={min(train_accs):.4f}, max={max(train_accs):.4f}, mean={sum(train_accs)/len(train_accs):.4f}")
    logger.info(f"Dev accuracy: min={min(dev_accs):.4f}, max={max(dev_accs):.4f}, mean={sum(dev_accs)/len(dev_accs):.4f}")

    # Filter by alpha threshold
    filtered = [p for p in programs if p["train_accuracy"] >= args.alpha]
    logger.info(f"Filtered to {len(filtered)}/{len(programs)} programs (alpha={args.alpha})")

    if not filtered:
        logger.warning("No programs passed filter! Using all programs ranked by dev accuracy.")
        filtered = programs

    # Rank by dev accuracy
    ranked = sorted(filtered, key=lambda p: p["dev_accuracy"], reverse=True)

    # Select top-k
    top_k = ranked[:args.k]

    logger.info(f"Selected top-{args.k} programs:")
    for i, p in enumerate(top_k):
        logger.info(
            f"  {i+1}. worker={p['worker_id']}, "
            f"source=problem_{p['source_schema']['problem_id']}{p['source_schema']['schema_key']}, "
            f"train_acc={p['train_accuracy']:.4f}, dev_acc={p['dev_accuracy']:.4f}"
        )

    # Prepare output
    output_data = {
        "theory": args.theory,
        "alpha": args.alpha,
        "k": args.k,
        "num_candidates": len(programs),
        "num_filtered": len(filtered),
        "programs": [
            {
                "code": p["code"],
                "source_schema": p["source_schema"],
                "train_accuracy": p["train_accuracy"],
                "dev_accuracy": p["dev_accuracy"],
            }
            for p in top_k
        ]
    }

    save_json(output_path, output_data)
    logger.info(f"Saved selected programs to {output_path}")


if __name__ == "__main__":
    main()
