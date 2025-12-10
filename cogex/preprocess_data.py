"""
COGEX Data Preprocessing Script

Transforms raw data.json into prompt/completion format for completion-only loss training.

Usage:
    python cogex/preprocess_data.py --input data/data.json --output-dir data/ --val-size 2000

Output format (for TRL DataCollatorForCompletionOnlyLM):
    prompt: "### Instruction\n{instruction}\n### Input\n{input}\n"
    completion: "### Code\n{code_prompt}\n### Output\n{code_output}"
"""

import json
import random
import argparse
from pathlib import Path


def format_prompt(example: dict) -> str:
    """Format the instruction portion (will be masked from loss)."""
    return f"### Instruction\n{example['instruction']}\n### Input\n{example['input']}\n"


def format_completion(example: dict) -> str:
    """Format the completion portion (loss will be computed on this)."""
    return f"### Code\n{example['code_prompt']}\n### Output\n{example['code_output']}"


def format_full_text(example: dict) -> str:
    """Format the full text for training."""
    return format_prompt(example) + format_completion(example)


def process_example(example: dict, idx: int) -> dict:
    """Transform raw example to prompt/completion format."""
    return {
        'id': example.get('id', str(idx)),
        'prompt': format_prompt(example),
        'completion': format_completion(example),
        'text': format_full_text(example),
    }


def validate_example(example: dict, idx: int) -> tuple[bool, str]:
    """Validate a single example for required fields and format."""
    required_fields = ['instruction', 'code_prompt', 'code_output']

    for field in required_fields:
        if field not in example:
            return False, f"Example {idx}: Missing field '{field}'"
        if not example[field] or not str(example[field]).strip():
            return False, f"Example {idx}: Empty field '{field}'"

    if '>>>' not in example['code_prompt']:
        return False, f"Example {idx}: Missing function call marker (>>>) in code_prompt"

    return True, ""


def compute_stats(examples: list[dict], tokenizer_name: str | None = None) -> dict:
    """Compute dataset statistics."""
    prompt_lens = [len(ex['prompt']) for ex in examples]
    completion_lens = [len(ex['completion']) for ex in examples]
    total_lens = [len(ex['text']) for ex in examples]

    stats = {
        'num_examples': len(examples),
        'prompt_char_len': {
            'min': min(prompt_lens),
            'max': max(prompt_lens),
            'mean': sum(prompt_lens) / len(prompt_lens),
        },
        'completion_char_len': {
            'min': min(completion_lens),
            'max': max(completion_lens),
            'mean': sum(completion_lens) / len(completion_lens),
        },
        'total_char_len': {
            'min': min(total_lens),
            'max': max(total_lens),
            'mean': sum(total_lens) / len(total_lens),
        },
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess COGEX dataset for finetuning'
    )
    parser.add_argument(
        '--input', type=str, default='data/data.json',
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--val-size', type=int, default=2000,
        help='Number of validation examples'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only verify data format, do not write output'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("COGEX Data Preprocessing")
    print("=" * 60)

    # Load raw data
    print(f"\n[1/5] Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        raw_data = json.load(f)
    print(f"      Loaded {len(raw_data):,} raw examples")

    # Validate examples
    print(f"\n[2/5] Validating examples...")
    valid_examples = []
    invalid_examples = []

    for i, ex in enumerate(raw_data):
        is_valid, error = validate_example(ex, i)
        if is_valid:
            valid_examples.append((i, ex))
        else:
            invalid_examples.append((i, error))

    print(f"      Valid: {len(valid_examples):,}")
    print(f"      Invalid: {len(invalid_examples):,}")

    if invalid_examples:
        print("\n      First 5 invalid examples:")
        for idx, error in invalid_examples[:5]:
            print(f"        - {error}")

    if args.verify_only:
        print("\n[--verify-only] Skipping output generation.")
        return

    # Shuffle and split
    print(f"\n[3/5] Shuffling and splitting (seed={args.seed})...")
    random.seed(args.seed)
    indices = list(range(len(valid_examples)))
    random.shuffle(indices)

    val_indices = set(indices[:args.val_size])
    train_indices = set(indices[args.val_size:])

    print(f"      Training set: {len(train_indices):,}")
    print(f"      Validation set: {len(val_indices):,}")

    # Process examples
    print(f"\n[4/5] Processing examples to prompt/completion format...")
    train_processed = []
    val_processed = []

    for i, (orig_idx, ex) in enumerate(valid_examples):
        processed = process_example(ex, orig_idx)
        if i in val_indices:
            val_processed.append(processed)
        else:
            train_processed.append(processed)

    # Compute statistics
    print(f"\n[5/5] Computing statistics...")
    train_stats = compute_stats(train_processed)
    val_stats = compute_stats(val_processed)

    print(f"\n      Training set statistics:")
    print(f"        Examples: {train_stats['num_examples']:,}")
    print(f"        Prompt length (chars): "
          f"min={train_stats['prompt_char_len']['min']}, "
          f"max={train_stats['prompt_char_len']['max']}, "
          f"mean={train_stats['prompt_char_len']['mean']:.1f}")
    print(f"        Completion length (chars): "
          f"min={train_stats['completion_char_len']['min']}, "
          f"max={train_stats['completion_char_len']['max']}, "
          f"mean={train_stats['completion_char_len']['mean']:.1f}")
    print(f"        Total length (chars): "
          f"min={train_stats['total_char_len']['min']}, "
          f"max={train_stats['total_char_len']['max']}, "
          f"mean={train_stats['total_char_len']['mean']:.1f}")

    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'train.json'
    val_path = output_dir / 'val.json'
    stats_path = output_dir / 'stats.json'

    print(f"\n      Saving to {output_dir}/...")

    with open(train_path, 'w') as f:
        json.dump(train_processed, f, indent=2)
    print(f"        train.json ({len(train_processed):,} examples)")

    with open(val_path, 'w') as f:
        json.dump(val_processed, f, indent=2)
    print(f"        val.json ({len(val_processed):,} examples)")

    full_stats = {
        'preprocessing': {
            'input_file': args.input,
            'seed': args.seed,
            'val_size': args.val_size,
            'total_raw': len(raw_data),
            'valid': len(valid_examples),
            'invalid': len(invalid_examples),
        },
        'train': train_stats,
        'val': val_stats,
    }

    with open(stats_path, 'w') as f:
        json.dump(full_stats, f, indent=2)
    print(f"        stats.json")

    # Print sample for verification
    print("\n" + "=" * 60)
    print("Sample processed example (first training example):")
    print("=" * 60)
    sample = train_processed[0]
    print(f"\n--- PROMPT (masked from loss) ---")
    print(sample['prompt'][:500] + "..." if len(sample['prompt']) > 500 else sample['prompt'])
    print(f"\n--- COMPLETION (loss computed) ---")
    print(sample['completion'][:500] + "..." if len(sample['completion']) > 500 else sample['completion'])

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
