"""
Evaluate a HuggingFace LLM on the FDT decision theory dataset.

Supports batch inference, SLURM parallelization, and checkpointing.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from llm import LLM

import torch
from transformers import (
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str | Path) -> list[dict]:
    """Load the FDT dataset from JSON."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data["problems"]


def save_json(path: Path, data: dict) -> None:
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def underline(text: str) -> str:
    """Convert to lowercase with underscores, remove non-word chars."""
    text = re.sub(r'[\s-]+', '_', text.strip().lower())
    return re.sub(r'[^\w]', '', text)


def extract_first_dict(text: str) -> str | None:
    """Extract the first balanced {...} from text."""
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def partition_problems(
    problems: list[dict],
    worker_id: int,
    num_workers: int,
) -> list[dict]:
    """Partition problems for this worker using modulo distribution."""
    return [p for p in problems if p["id"] % num_workers == worker_id]


def pad_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
    padding_side: str = "left",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences to the same length and create attention masks.

    Returns:
        input_ids: (batch_size, max_len) tensor
        attention_mask: (batch_size, max_len) tensor
    """
    max_len = max(len(seq) for seq in sequences)

    padded = []
    masks = []

    for seq in sequences:
        pad_len = max_len - len(seq)
        if padding_side == "left":
            padded_seq = [pad_token_id] * pad_len + seq
            mask = [0] * pad_len + [1] * len(seq)
        else:
            padded_seq = seq + [pad_token_id] * pad_len
            mask = [1] * len(seq) + [0] * pad_len
        padded.append(padded_seq)
        masks.append(mask)

    return torch.tensor(padded), torch.tensor(masks)


def cogex_formatter(schema: dict, tokenizer: PreTrainedTokenizerBase) -> str:
    """Format a schema for COGEX-style evaluation."""
    return (f"### Instruction\nReturn the action which would maximize your "
            f"personal expected value in this game theory problem.\n"
            f"### Input\n{schema['description']}\n"
            f"actions: {underline(schema['a'][0])}, {underline(schema['a'][1])}\n"
            f"### Code\n")


def chat_formatter(schema: dict, tokenizer: PreTrainedTokenizerBase) -> str:
    """Format a schema for chat-style evaluation."""
    content = (f"{schema['description']}\n\nReturn the action which would "
               f"maximize your personal expected value in this game theory "
               f"problem.\nactions: {underline(schema['a'][0])}, "
               f"{underline(schema['a'][1])} /no_think")
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def cogex_parser(schema: dict, output: str, use_llm_fallback: bool = True) -> int:
    """Parse COGEX-style output to extract the choice."""
    # First try hardcoded parsing
    output_match = re.search(r"### Output", output)
    if output_match:
        after_output = output[output_match.end():]
        dict_str = extract_first_dict(after_output)
        if dict_str:
            try:
                output_dict = ast.literal_eval(dict_str)
                answer = output_dict.get('answer', -1)
                if isinstance(answer, str):
                    if underline(answer) == underline(schema['a'][0]):
                        return 0
                    elif underline(answer) == underline(schema['a'][1]):
                        return 1
            except (SyntaxError, ValueError):
                pass

    # Fallback to LLM parser
    if use_llm_fallback:
        return llm_fallback_parser(output, schema['a'][0], schema['a'][1])
    return -1


# Global for lazy-loaded parser LLM
_parser_llm: LLM | None = None


def get_parser_llm() -> LLM:
    """Lazy-load Qwen3-4B-Instruct for fallback parsing."""
    global _parser_llm
    if _parser_llm is None:
        from dotenv import load_dotenv
        from llm import LLM
        load_dotenv()
        models_path = os.getenv("MODELS_PATH", "")
        model_path = os.path.join(models_path, "Qwen3-4B-Instruct")
        logger.info(f"Loading fallback parser LLM from {model_path}")
        _parser_llm = LLM(
            model_path,
            model_kwargs={
                "dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
            generation_defaults={"max_new_tokens": 16},
        )
    return _parser_llm


def llm_fallback_parser(output_text: str, action_0: str, action_1: str) -> int:
    """Use Qwen3-4B-Instruct to parse ambiguous outputs."""
    # Extract output dict if present (COGEX format), otherwise use full text (chat format)
    output_match = re.search(r'output\s*=\s*(\{.*\})', output_text, re.DOTALL)
    if output_match:
        text_to_parse = output_match.group(1)[:1000]
    else:
        text_to_parse = output_text[:1000]

    prompt = f"""Given this output from a decision model:
{text_to_parse}

The two actions were:
Action 0: {action_0}
Action 1: {action_1}

Which action was chosen? Reply with a single token: 0, 1, or -1 (if unclear). The wording does not need to be exact, as long as it is obvious which action they intended to say."""

    llm = get_parser_llm()
    response = llm.chat(prompt, use_history=False, save_history=False)

    # Parse response
    response = response.strip()
    if response.startswith("0"):
        return 0
    elif response.startswith("1"):
        return 1
    return -1


def chat_parser(schema: dict, output: str, use_llm_fallback: bool = True) -> int:
    """Parse chat-style output to extract the choice."""
    # Strip out chain of thought
    output = re.sub(r'<think>.+</think>', '', output, flags=re.DOTALL)

    action0 = underline(schema['a'][0])
    action1 = underline(schema['a'][1])

    pos0 = output.rfind(action0)
    pos1 = output.rfind(action1)

    if pos0 == -1 and pos1 == -1:
        # Fallback to LLM parser
        if use_llm_fallback:
            return llm_fallback_parser(output, schema['a'][0], schema['a'][1])
        return -1
    elif pos0 == -1:
        return 1
    elif pos1 == -1:
        return 0
    else:
        return 0 if pos0 < pos1 else 1


def evaluate_fdt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    formatter: Callable[[dict, PreTrainedTokenizerBase], str],
    parser: Callable[[dict, str], int],
    dataset_path: str | Path = "data/fdt.json",
    output_dir: str | Path = "data/tmp",
    batch_size: int = 16,
    checkpoint_every: int = 10,
    worker_id: int = 0,
    num_workers: int = 1,
    add_special_tokens: bool = False,
    max_new_tokens: int = 1024,
    retry_max_new_tokens: int = 2048,
    **generation_kwargs: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Evaluate a model on the FDT decision theory dataset.

    Args:
        model: HuggingFace model (AutoModelForCausalLM or PeftModelForCausalLM)
        tokenizer: HuggingFace tokenizer
        formatter: Function that takes schema dict and tokenizer, returns str
        parser: Function that takes schema dict and output str, returns 0, 1, or -1
        dataset_path: Path to fdt.json
        output_dir: Directory for checkpoints and results
        batch_size: Number of prompts to process at once
        checkpoint_every: Save checkpoint after this many problems
        worker_id: This worker's ID (0-indexed)
        num_workers: Total number of workers
        add_special_tokens: Whether to add special tokens during tokenization
        **generation_kwargs: Additional kwargs for model.generate()

    Returns:
        Dict mapping problem_id -> {"A": {"choice": int, "text": str}, ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Load dataset and partition for this worker
    problems = load_dataset(dataset_path)
    my_problems = partition_problems(problems, worker_id, num_workers)
    logger.info(f"Worker {worker_id}/{num_workers}: Processing {len(my_problems)} problems")

    # Load checkpoint if exists
    checkpoint_path = output_dir / f"checkpoint_{worker_id}.json"
    if checkpoint_path.exists():
        completed = load_json(checkpoint_path)
        skip_ids = set(completed.keys())
        logger.info(f"Resuming from checkpoint with {len(completed)} completed problems")
    else:
        completed: dict[str, dict[str, dict[str, Any]]] = {}
        skip_ids: set[str] = set()

    # Set up generation defaults
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"

    # If model is on a single accelerator, move inputs to that device
    generation_device: torch.device | None = None
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        unique_devices = {str(d) for d in device_map.values()}
        if len(unique_devices) == 1 and not {"cpu", "disk"} & unique_devices:
            single = next(iter(unique_devices))
            generation_device = torch.device(f"cuda:{single}") if single.isdigit() else torch.device(single)
    else:
        try:
            param_device = next(model.parameters()).device
            if param_device.type != "cpu":
                generation_device = param_device
        except StopIteration:
            generation_device = None

    gen_defaults = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    user_logits_processor = generation_kwargs.pop("logits_processor", None)
    base_logits_processors = []
    if user_logits_processor:
        if isinstance(user_logits_processor, LogitsProcessorList):
            base_logits_processors.extend(list(user_logits_processor))
        elif isinstance(user_logits_processor, list):
            base_logits_processors.extend(user_logits_processor)
        else:
            base_logits_processors.append(user_logits_processor)
    gen_defaults.update(generation_kwargs)

    # Collect all tasks: (problem_id, schema_key, schema_data, input_ids)
    tasks: list[tuple[str, str, dict, list[int]]] = []

    for problem in my_problems:
        problem_id = str(problem["id"])
        if problem_id in skip_ids:
            continue

        for schema_key in ["A", "B", "C"]:
            schema_data = problem["schema"][schema_key]
            formatted = formatter(schema_data, tokenizer)
            tokens = tokenizer.encode(formatted, add_special_tokens=add_special_tokens)
            tasks.append((problem_id, schema_key, schema_data, tokens))

    logger.info(f"Total tasks to process: {len(tasks)}")

    # Process in batches
    results_buffer: dict[str, dict[str, dict[str, Any]]] = {}
    problems_since_checkpoint = 0

    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        # Pad batch
        batch_tokens = [t[3] for t in batch_tasks]
        input_ids, attention_mask = pad_sequences(
            batch_tokens,
            tokenizer.pad_token_id,
            padding_side="left",
        )
        if generation_device is not None:
            input_ids = input_ids.to(generation_device)
            attention_mask = attention_mask.to(generation_device)

        batch_generate_kwargs = dict(gen_defaults)
        if base_logits_processors:
            batch_generate_kwargs["logits_processor"] = LogitsProcessorList(base_logits_processors)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **batch_generate_kwargs,
            )

        # Decode and parse outputs
        prompt_length = input_ids.shape[-1]

        for i, (problem_id, schema_key, schema_data, _) in enumerate(batch_tasks):
            completion_tokens = output_ids[i, prompt_length:]
            output_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)

            choice = parser(schema_data, output_text)

            if choice not in (0, 1):
                logger.warning(
                    f"Parser returned {choice} for problem {problem_id} schema {schema_key}.\n"
                    f"Output:\n{output_text}\n{'='*60}"
                )

            if problem_id not in results_buffer:
                results_buffer[problem_id] = {}

            results_buffer[problem_id][schema_key] = {
                "choice": choice,
                "text": output_text,
            }

            # Check if problem is complete
            if len(results_buffer[problem_id]) == 3:
                completed[problem_id] = results_buffer.pop(problem_id)
                problems_since_checkpoint += 1

                # Checkpoint
                if problems_since_checkpoint >= checkpoint_every:
                    save_json(checkpoint_path, completed)
                    logger.info(f"Checkpoint saved: {len(completed)} problems completed")
                    problems_since_checkpoint = 0

        logger.info(f"Processed batch {batch_start//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")

    # Retry failed schemas with more tokens
    failed_tasks: list[tuple[str, str, dict, list[int]]] = []
    for problem_id, schemas in completed.items():
        for schema_key, result in schemas.items():
            if result["choice"] == -1:
                # Find the original problem to get schema data
                for problem in my_problems:
                    if str(problem["id"]) == problem_id:
                        schema_data = problem["schema"][schema_key]
                        formatted = formatter(schema_data, tokenizer)
                        tokens = tokenizer.encode(formatted, add_special_tokens=add_special_tokens)
                        failed_tasks.append((problem_id, schema_key, schema_data, tokens))
                        break

    if failed_tasks:
        logger.info(f"Retrying {len(failed_tasks)} failed schemas with {retry_max_new_tokens} tokens")
        retry_gen_defaults = dict(gen_defaults)
        retry_gen_defaults["max_new_tokens"] = retry_max_new_tokens

        for batch_start in range(0, len(failed_tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(failed_tasks))
            batch_tasks = failed_tasks[batch_start:batch_end]

            batch_tokens = [t[3] for t in batch_tasks]
            input_ids, attention_mask = pad_sequences(
                batch_tokens,
                tokenizer.pad_token_id,
                padding_side="left",
            )
            if generation_device is not None:
                input_ids = input_ids.to(generation_device)
                attention_mask = attention_mask.to(generation_device)

            batch_generate_kwargs = dict(retry_gen_defaults)
            if base_logits_processors:
                batch_generate_kwargs["logits_processor"] = LogitsProcessorList(base_logits_processors)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **batch_generate_kwargs,
                )

            prompt_length = input_ids.shape[-1]

            for i, (problem_id, schema_key, schema_data, _) in enumerate(batch_tasks):
                completion_tokens = output_ids[i, prompt_length:]
                output_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)

                choice = parser(schema_data, output_text)

                if choice not in (0, 1):
                    logger.warning(
                        f"Retry still failed for problem {problem_id} schema {schema_key}.\n"
                        f"Output:\n{output_text}\n{'='*60}"
                    )

                # Update the result
                completed[problem_id][schema_key] = {
                    "choice": choice,
                    "text": output_text,
                }

            logger.info(f"Processed retry batch {batch_start//batch_size + 1}/{(len(failed_tasks) + batch_size - 1)//batch_size}")

    # Final save
    results_path = output_dir / f"results_{worker_id}.json"
    save_json(results_path, completed)
    logger.info(f"Final results saved to {results_path}: {len(completed)} problems")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed after successful completion")

    return completed


def main():
    """CLI entry point for SLURM jobs."""
    import argparse
    import os

    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate LLM on FDT dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="data/fdt.json", help="Path to fdt.json")
    parser.add_argument("--output-dir", type=str, default="data/tmp", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of workers")
    parser.add_argument("--lora-path", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument(
        "--model-type", type=str, choices=["cogex", "chat"], required=True,
        help="Model type: 'cogex' for COGEX-trained models, 'chat' for chat models"
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Force model onto a single GPU if available (avoids sharding/offload)",
    )

    args = parser.parse_args()

    # Resolve model path
    models_path = os.getenv("MODELS_PATH", "")
    if models_path and not os.path.isabs(args.model):
        model_path = os.path.join(models_path, args.model)
    else:
        model_path = args.model

    logger.info(f"Loading model from {model_path}")

    use_single_gpu = args.single_gpu and torch.cuda.is_available()
    if args.single_gpu and not torch.cuda.is_available():
        logger.warning("single-gpu requested but CUDA not available; falling back to auto device_map")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None if use_single_gpu else "auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    if use_single_gpu:
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load LoRA adapter if specified
    if args.lora_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)

    # Select formatter, parser, and token limits based on model type
    if args.model_type == "cogex":
        formatter = cogex_formatter
        output_parser = cogex_parser
        max_new_tokens = 1024
        retry_max_new_tokens = 2048
    else:  # chat
        formatter = chat_formatter
        output_parser = chat_parser
        max_new_tokens = 1024
        retry_max_new_tokens = 2048

    # Run evaluation
    evaluate_fdt(
        model=model,
        tokenizer=tokenizer,
        formatter=formatter,
        parser=output_parser,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        add_special_tokens=(args.model_type == "cogex"),
        max_new_tokens=max_new_tokens,
        retry_max_new_tokens=retry_max_new_tokens,
    )


if __name__ == "__main__":
    main()
