"""Merge fdt-ABC and fdt-description files into a single fdt.json.

Matches entries by theme (not by file index, since those don't correlate).
Adds a 'description' field to each A, B, C topology in the schema.
"""

from glob import glob
import argparse
import json


def load_abc_files(pattern: str) -> dict[str, dict]:
    """Load all ABC files and index by theme."""
    abc_by_theme: dict[str, dict] = {}
    duplicates = []

    for fp in glob(pattern):
        with open(fp) as f:
            data = json.load(f)

        for problem in data["problems"]:
            theme = problem["theme"]
            if theme in abc_by_theme:
                duplicates.append(theme)
            else:
                abc_by_theme[theme] = problem

    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate themes in ABC files")

    return abc_by_theme


def load_description_files(pattern: str) -> dict[str, dict]:
    """Load all description files and index by theme."""
    desc_by_theme: dict[str, dict] = {}
    duplicates = []

    for fp in glob(pattern):
        with open(fp) as f:
            data = json.load(f)

        for desc in data["descriptions"]:
            theme = desc["theme"]
            if theme in desc_by_theme:
                duplicates.append(theme)
            else:
                desc_by_theme[theme] = desc

    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate themes in description files")

    return desc_by_theme


def merge_datasets(
    abc_by_theme: dict[str, dict],
    desc_by_theme: dict[str, dict],
    include_unmatched: bool = False,
) -> list[dict]:
    """Merge ABC problems with descriptions by theme."""
    merged = []
    matched = 0
    unmatched_abc = 0
    id_counter = 0

    for theme, problem in abc_by_theme.items():
        if theme in desc_by_theme:
            desc = desc_by_theme[theme]
            # Add description to each topology
            for topology in ["A", "B", "C"]:
                problem["schema"][topology]["description"] = desc[topology]
            problem["id"] = id_counter
            id_counter += 1
            merged.append(problem)
            matched += 1
        elif include_unmatched:
            problem["id"] = id_counter
            id_counter += 1
            merged.append(problem)
            unmatched_abc += 1
        else:
            unmatched_abc += 1

    # Check for orphaned descriptions
    orphaned = set(desc_by_theme.keys()) - set(abc_by_theme.keys())

    print(f"ABC problems loaded: {len(abc_by_theme)}")
    print(f"Descriptions loaded: {len(desc_by_theme)}")
    print(f"Successfully matched: {matched}")
    print(f"ABC without description: {unmatched_abc}")
    print(f"Orphaned descriptions: {len(orphaned)}")

    if orphaned:
        print("Orphaned themes (first 5):")
        for theme in list(orphaned)[:5]:
            print(f"  - {theme[:60]}...")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge fdt-ABC and fdt-description files into fdt.json"
    )
    parser.add_argument(
        "--abc-pattern",
        default="data/tmp/fdt-ABC-*.json",
        help="Glob pattern for ABC files",
    )
    parser.add_argument(
        "--desc-pattern",
        default="data/tmp/fdt-description-*.json",
        help="Glob pattern for description files",
    )
    parser.add_argument(
        "--output",
        default="data/fdt.json",
        help="Output file path",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Include ABC problems without descriptions",
    )
    args = parser.parse_args()

    abc_by_theme = load_abc_files(args.abc_pattern)
    desc_by_theme = load_description_files(args.desc_pattern)

    merged = merge_datasets(abc_by_theme, desc_by_theme, args.include_unmatched)

    output = {"problems": merged}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to {args.output} ({len(merged)} problems)")


if __name__ == "__main__":
    main()
