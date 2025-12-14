"""
CoTACS program search - Phase 1.

Each worker:
1. Generates ONE program from its assigned train schema (temp=0.7)
2. Evaluates that program on ALL train schemas
3. Evaluates that program on ALL dev schemas
4. Saves program + scores

Usage:
    python -m cotacs.search \
        --model Llama-2-7b-hf \
        --lora-path outputs/cogex-.../checkpoint-7775 \
        --theory FDT \
        --splits data/cotacs/splits.json \
        --output-dir data/cotacs/FDT \
        --worker-id 0 \
        --num-workers 33
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
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
    cogex_formatter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DecisionTheory = Literal["CDT", "FDT", "EDT"]


def extract_function_def(output: str) -> tuple[str, str]:
    """
    Extract the function definition and name from a COGEX generation output.

    COGEX generation produces:
        def func_name(...):
            ...
        >>> func_name(...)
        ### Output
        {...}

    This extracts only the function definition (before ">>>"),
    NOT the specific function call which has instance-specific arguments.

    Returns:
        (function_definition, function_name)
    """
    # Find the >>> line which marks the function invocation
    lines = output.split('\n')
    code_lines = []
    for line in lines:
        if line.strip().startswith('>>>'):
            break
        code_lines.append(line)

    code = '\n'.join(code_lines).strip()

    # Fallback: if no >>> found, try ### Output marker
    if not code or '>>>' not in output:
        marker = "### Output"
        idx = output.find(marker)
        if idx != -1:
            code = output[:idx].strip()
        else:
            code = output.strip()

    # Extract function name from "def func_name(..."
    func_name = "solve"  # default fallback
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        func_name = match.group(1)

    return code, func_name


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

    Args:
        schema: Schema dict with description, actions
        program_code: The fixed function definition (no >>> call)

    Returns:
        Formatted prompt ending after the function definition
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


def get_schemas_for_ids(
    problems: list[dict],
    problem_ids: list[int],
    theory: DecisionTheory,
) -> list[dict]:
    """
    Get all schemas for given problem IDs with ground truth labels.

    Returns list of dicts with: problem_id, schema_key, schema (data), ground_truth
    """
    schemas = []
    id_set = set(problem_ids)
    for problem in problems:
        if problem["id"] in id_set:
            for schema_key in ["A", "B", "C"]:
                schema_data = problem["schema"][schema_key]
                schemas.append({
                    "problem_id": problem["id"],
                    "schema_key": schema_key,
                    "schema": schema_data,
                    "ground_truth": schema_data[theory],
                })
    return schemas


def generate_program(
    model,
    tokenizer,
    schema: dict,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> tuple[str, str]:
    """
    Generate a candidate program from a schema.

    Args:
        model: COGEX model
        tokenizer: Tokenizer
        schema: Schema dict to generate from
        temperature: Sampling temperature (0.7 for diversity)
        max_new_tokens: Max tokens to generate

    Returns:
        (function_definition, function_name) - just the def, no >>> call
    """
    prompt = cogex_formatter(schema["schema"], tokenizer)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[-1]
    completion = output_ids[0, prompt_len:]
    output_text = tokenizer.decode(completion, skip_special_tokens=True)

    code, func_name = extract_function_def(output_text)
    return code, func_name


def evaluate_program(
    model,
    tokenizer,
    program_code: str,
    schemas: list[dict],
    batch_size: int = 8,
    temperature: float = 0.05,
    max_new_tokens: int = 256,
) -> float:
    """
    Evaluate a program on a set of schemas using emulation mode.

    Args:
        model: COGEX model
        tokenizer: Tokenizer
        program_code: Fixed program code
        schemas: List of schema dicts with ground_truth
        batch_size: Batch size for inference
        temperature: Low temperature for near-deterministic output
        max_new_tokens: Max tokens for output dict

    Returns:
        Accuracy (fraction correct)
    """
    correct = 0
    total = 0

    for batch_start in range(0, len(schemas), batch_size):
        batch_schemas = schemas[batch_start:batch_start + batch_size]

        # Format emulation prompts
        prompts = [
            format_emulation_prompt(s["schema"], program_code)
            for s in batch_schemas
        ]

        # Tokenize and pad
        token_lists = [
            tokenizer.encode(p, add_special_tokens=True)
            for p in prompts
        ]
        input_ids, attention_mask = pad_sequences(
            token_lists, tokenizer.pad_token_id, padding_side="left"
        )
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        # Generate
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

        # Parse and check accuracy
        prompt_len = input_ids.shape[-1]
        for i, schema in enumerate(batch_schemas):
            completion = output_ids[i, prompt_len:]
            output_text = tokenizer.decode(completion, skip_special_tokens=True)

            # Model generates ">>> call(...)\n### Output\n{...}"
            # cogex_parser looks for ### Output marker, so pass the full output
            choice = cogex_parser(schema["schema"], output_text, use_llm_fallback=False)

            if choice == schema["ground_truth"]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="CoTACS program search (Phase 1)"
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
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--worker-id", type=int, default=0,
        help="Worker ID (0-indexed)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=99,
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

    # Load splits and dataset
    splits = load_json(Path(args.splits))
    problems = load_dataset(args.dataset)

    train_ids = splits["train_ids"]
    dev_ids = splits["dev_ids"]

    # Get schemas
    train_schemas = get_schemas_for_ids(problems, train_ids, args.theory)
    dev_schemas = get_schemas_for_ids(problems, dev_ids, args.theory)

    logger.info(f"Train schemas: {len(train_schemas)}, Dev schemas: {len(dev_schemas)}")

    # Determine which train schema this worker handles
    if args.worker_id >= len(train_schemas):
        logger.warning(f"Worker {args.worker_id} has no work (only {len(train_schemas)} train schemas)")
        return

    my_schema = train_schemas[args.worker_id]
    logger.info(
        f"Worker {args.worker_id}: Generating program from problem {my_schema['problem_id']} "
        f"schema {my_schema['schema_key']}"
    )

    # Step 1: Generate program
    logger.info("Step 1: Generating candidate program...")
    program_code, func_name = generate_program(model, tokenizer, my_schema, temperature=0.7)
    logger.info(f"Generated function '{func_name}':\n{program_code[:500]}...")

    # Step 2: Evaluate on train set
    logger.info("Step 2: Evaluating on train set...")
    train_accuracy = evaluate_program(
        model, tokenizer, program_code, train_schemas,
        batch_size=args.batch_size, temperature=0.05
    )
    logger.info(f"Train accuracy: {train_accuracy:.4f}")

    # Step 3: Evaluate on dev set
    logger.info("Step 3: Evaluating on dev set...")
    dev_accuracy = evaluate_program(
        model, tokenizer, program_code, dev_schemas,
        batch_size=args.batch_size, temperature=0.05
    )
    logger.info(f"Dev accuracy: {dev_accuracy:.4f}")

    # Save results
    output_dir = Path(args.output_dir) / args.theory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"search_{args.worker_id}.json"

    result = {
        "worker_id": args.worker_id,
        "source_schema": {
            "problem_id": my_schema["problem_id"],
            "schema_key": my_schema["schema_key"],
        },
        "code": program_code,
        "func_name": func_name,
        "train_accuracy": train_accuracy,
        "dev_accuracy": dev_accuracy,
        "config": {
            "model": args.model,
            "lora_path": args.lora_path,
            "theory": args.theory,
            "num_train_schemas": len(train_schemas),
            "num_dev_schemas": len(dev_schemas),
        }
    }

    save_json(output_path, result)
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
