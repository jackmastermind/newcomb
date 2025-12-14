"""
CoTACS evaluation - Phase 3.

Evaluates selected programs on the eval set using emulation mode.
Supports parallelization via worker_id/num_workers.

Usage:
    python -m cotacs.eval \
        --model Llama-2-7b-hf \
        --lora-path outputs/cogex-.../checkpoint-7775 \
        --theory FDT \
        --programs data/cotacs/FDT/programs.json \
        --splits data/cotacs/splits.json \
        --output-dir data/cotacs/FDT \
        --worker-id 0 \
        --num-workers 52
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_fdt import (
    load_dataset,
    save_json,
    load_json,
    underline,
    pad_sequences,
    cogex_parser,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DecisionTheory = Literal["CDT", "FDT", "EDT"]


def format_emulation_prompt(
    schema: dict,
    program_code: str,
) -> str:
    """
    Format prompt for emulation mode (includes fixed program code).

    The COGEX model is trained to generate:
    1. The >>> function_call(...) with appropriate arguments from ### Input
    2. The ### Output section with the result

    So we just provide the function definition after ### Code.
    """
    return (
        f"### Instruction\n"
        f"Return the action which would maximize your personal expected value "
        f"in this game theory problem.\n"
        f"### Input\n"
        f"{schema['description']}\n"
        f"actions: {underline(schema['a'][0])}, {underline(schema['a'][1])}\n"
        f"### Code\n"
        f"{program_code}\n"
    )


def partition_problems(
    problems: list[dict],
    eval_ids: list[int],
    worker_id: int,
    num_workers: int,
) -> list[dict]:
    """Partition eval problems for this worker using modulo distribution."""
    eval_id_set = set(eval_ids)
    eval_problems = [p for p in problems if p["id"] in eval_id_set]
    return [p for p in eval_problems if p["id"] % num_workers == worker_id]


def evaluate_with_programs(
    model,
    tokenizer,
    programs: list[dict],
    problem: dict,
    theory: DecisionTheory,
    batch_size: int = 8,
    temperature: float = 0.05,
    max_new_tokens: int = 256,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate a single problem with all programs using emulation mode.

    Returns dict mapping schema_key -> {choice, text, ground_truth}
    """
    results = {}

    for schema_key in ["A", "B", "C"]:
        schema = problem["schema"][schema_key]
        ground_truth = schema[theory]

        if len(programs) == 1:
            # Single program - direct evaluation
            program_code = programs[0]["code"]
            prompt = format_emulation_prompt(schema, program_code)

            input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            input_ids = input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = input_ids.shape[-1]
            completion = output_ids[0, prompt_len:]
            output_text = tokenizer.decode(completion, skip_special_tokens=True)

            # Model generates ">>> call(...)\n### Output\n{...}"
            # cogex_parser looks for ### Output marker, so pass the full output
            choice = cogex_parser(schema, output_text, use_llm_fallback=False)

            results[schema_key] = {
                "choice": choice,
                "text": output_text,
                "ground_truth": ground_truth,
            }
        else:
            # Multiple programs - majority vote
            votes = []
            texts = []

            for program in programs:
                program_code = program["code"]
                prompt = format_emulation_prompt(schema, program_code)

                input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
                input_ids = input_ids.to(model.device)
                attention_mask = torch.ones_like(input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                prompt_len = input_ids.shape[-1]
                completion = output_ids[0, prompt_len:]
                output_text = tokenizer.decode(completion, skip_special_tokens=True)

                # Model generates ">>> call(...)\n### Output\n{...}"
                # cogex_parser looks for ### Output marker, so pass the full output
                choice = cogex_parser(schema, output_text, use_llm_fallback=False)
                votes.append(choice)
                texts.append(output_text)

            # Majority vote (excluding -1 if possible)
            valid_votes = [v for v in votes if v in (0, 1)]
            if valid_votes:
                final_choice = Counter(valid_votes).most_common(1)[0][0]
            else:
                final_choice = -1

            results[schema_key] = {
                "choice": final_choice,
                "text": texts[0],  # Keep first program's text for reference
                "ground_truth": ground_truth,
                "votes": votes,
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CoTACS evaluation (Phase 3)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name or path"
    )
    parser.add_argument(
        "--lora-path", type=str, default=None,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--theory", type=str, required=True, choices=["CDT", "FDT", "EDT"],
        help="Decision theory for ground truth labels"
    )
    parser.add_argument(
        "--programs", type=str, required=True,
        help="Path to programs.json from select"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/fdt.json",
        help="Path to fdt.json"
    )
    parser.add_argument(
        "--splits", type=str, default="data/cotacs/splits.json",
        help="Path to splits.json"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/cotacs",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Checkpoint frequency (problems)"
    )
    parser.add_argument(
        "--worker-id", type=int, default=0,
        help="Worker ID (0-indexed)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Total number of workers"
    )

    args = parser.parse_args()

    # Load dotenv for MODELS_PATH
    from dotenv import load_dotenv
    load_dotenv()

    # Resolve model path
    models_path = os.getenv("MODELS_PATH", "")
    if models_path and not os.path.isabs(args.model):
        model_path = os.path.join(models_path, args.model)
    else:
        model_path = args.model

    logger.info(f"Loading model from {model_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load LoRA adapter
    if args.lora_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()

    # Load programs, splits, and dataset
    programs_data = load_json(Path(args.programs))
    programs = programs_data["programs"]
    logger.info(f"Loaded {len(programs)} programs for evaluation")

    splits = load_json(Path(args.splits))
    eval_ids = splits["eval_ids"]

    problems = load_dataset(args.dataset)
    my_problems = partition_problems(problems, eval_ids, args.worker_id, args.num_workers)

    logger.info(
        f"Worker {args.worker_id}/{args.num_workers}: "
        f"Processing {len(my_problems)} problems"
    )

    # Output setup
    output_dir = Path(args.output_dir) / args.theory
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{args.worker_id}.json"
    results_path = output_dir / f"results_{args.worker_id}.json"

    # Load checkpoint if exists
    if checkpoint_path.exists():
        completed = load_json(checkpoint_path)
        skip_ids = set(completed.keys())
        logger.info(f"Resuming from checkpoint with {len(completed)} completed problems")
    else:
        completed: dict[str, dict] = {}
        skip_ids: set[str] = set()

    # Evaluate
    problems_since_checkpoint = 0
    for problem in my_problems:
        problem_id = str(problem["id"])
        if problem_id in skip_ids:
            continue

        results = evaluate_with_programs(
            model, tokenizer, programs, problem, args.theory,
            batch_size=args.batch_size, temperature=0.05
        )

        completed[problem_id] = results
        problems_since_checkpoint += 1

        # Checkpoint
        if problems_since_checkpoint >= args.checkpoint_every:
            save_json(checkpoint_path, completed)
            logger.info(f"Checkpoint saved: {len(completed)} problems completed")
            problems_since_checkpoint = 0

    # Final save
    save_json(results_path, completed)
    logger.info(f"Final results saved to {results_path}: {len(completed)} problems")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed after successful completion")


if __name__ == "__main__":
    main()
