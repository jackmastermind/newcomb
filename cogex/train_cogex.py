"""
COGEX Finetuning Script

Train LLMs to generate pseudo-code and emulate execution using LoRA.

Usage:
    python cogex/train_cogex.py \
        --model Llama-3.1-8B \
        --output-dir outputs/cogex-llama3.1-8b \
        --wandb-project cogex-replication

Environment:
    MODELS_PATH: Directory containing model weights (from .env)
    WANDB_API_KEY: Weights & Biases API key (optional, for logging)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


@dataclass
class TrainingArgs:
    """All training arguments in one place."""
    # Model
    model_name: str = "Llama-3.1-8B"
    attn_implementation: str = "auto"  # auto, flash_attention_2, sdpa, eager

    # Data
    train_file: str = "data/train.json"
    val_file: str = "data/val.json"

    # Output
    output_dir: str = "outputs/cogex"

    # Training hyperparameters (from COGEX paper)
    num_train_epochs: int = 5
    learning_rate: float = 3e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # LoRA configuration (from COGEX paper)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Memory optimization
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 5

    # WandB
    wandb_project: str = "cogex-replication"
    wandb_run_name: str | None = None
    use_wandb: bool = True

    # Misc
    seed: int = 42


def load_env() -> str:
    """Load environment and return MODELS_PATH."""
    load_dotenv()
    models_path = os.getenv('MODELS_PATH')
    if not models_path:
        print("ERROR: MODELS_PATH not set in environment or .env file")
        sys.exit(1)
    return models_path


def detect_attention_implementation(model_path: str) -> str:
    """Auto-detect the best available attention implementation."""
    print("\n[Attention] Auto-detecting best implementation...")

    # Try flash_attention_2 first
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            print("[Attention] flash_attention_2 is available")
            return "flash_attention_2"
        else:
            print("[Attention] flash_attention_2 not available")
    except ImportError:
        print("[Attention] flash_attn package not installed")

    # Fall back to SDPA (PyTorch native, still efficient)
    print("[Attention] Using sdpa (PyTorch scaled dot product attention)")
    return "sdpa"


def load_datasets(args: TrainingArgs) -> tuple[Dataset, Dataset]:
    """Load preprocessed train and validation datasets."""
    print(f"\n[Data] Loading datasets...")
    print(f"  Train: {args.train_file}")
    print(f"  Val: {args.val_file}")

    with open(args.train_file, 'r') as f:
        train_data = json.load(f)

    with open(args.val_file, 'r') as f:
        val_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    print(f"  Train size: {len(train_dataset):,}")
    print(f"  Val size: {len(val_dataset):,}")

    return train_dataset, val_dataset


def load_model_and_tokenizer(
    args: TrainingArgs,
    models_path: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and tokenizer with proper configuration."""
    model_path = Path(models_path) / args.model_name

    print(f"\n[Model] Loading from {model_path}")

    # Detect attention implementation
    if args.attn_implementation == "auto":
        attn_impl = detect_attention_implementation(str(model_path))
    else:
        attn_impl = args.attn_implementation
        print(f"[Attention] Using specified: {attn_impl}")

    # Load tokenizer
    print("[Model] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Model] Set pad_token to eos_token: {tokenizer.pad_token}")
    tokenizer.padding_side = "right"

    # Load model
    print("[Model] Loading model weights (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=attn_impl,
        trust_remote_code=False,
        device_map="auto",
    )

    print(f"[Model] Loaded successfully")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  dtype: {model.dtype}")

    return model, tokenizer


def setup_lora(model: AutoModelForCausalLM, args: TrainingArgs) -> AutoModelForCausalLM:
    """Apply LoRA adapters to the model."""
    print(f"\n[LoRA] Applying adapters...")
    print(f"  r: {args.lora_r}")
    print(f"  alpha: {args.lora_alpha}")
    print(f"  dropout: {args.lora_dropout}")
    print(f"  targets: {args.lora_target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def run_sanity_checks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: TrainingArgs,
) -> bool:
    """Run comprehensive pre-training sanity checks."""
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    all_passed = True

    # Check 1: Dataset sizes
    print(f"\n[1] Dataset sizes")
    print(f"    Train: {len(train_dataset):,}")
    print(f"    Val: {len(val_dataset):,}")
    if len(train_dataset) < 1000:
        print("    WARNING: Training set seems small")

    # Check 2: Sample data format
    print(f"\n[2] Data format")
    sample = train_dataset[0]
    required_keys = ['text', 'prompt', 'completion']
    for key in required_keys:
        if key in sample:
            print(f"    {key}: present ({len(sample[key])} chars)")
        else:
            print(f"    {key}: MISSING!")
            all_passed = False

    # Check 3: Response template in data
    print(f"\n[3] Response template detection")
    response_template = "### Code"
    n_with_template = sum(1 for ex in train_dataset if response_template in ex.get('text', ''))
    print(f"    Template '{response_template}' found in {n_with_template:,}/{len(train_dataset):,} examples")
    if n_with_template < len(train_dataset) * 0.99:
        print("    WARNING: Some examples missing response template!")
        all_passed = False

    # Check 4: Tokenization
    print(f"\n[4] Tokenization")
    sample_text = train_dataset[0]['text']
    tokens = tokenizer.encode(sample_text)
    print(f"    Sample token count: {len(tokens)}")

    # Check sequence length distribution
    print(f"    Checking sequence lengths (first 100 examples)...")
    lengths = []
    for i, ex in enumerate(train_dataset):
        if i >= 100:
            break
        lengths.append(len(tokenizer.encode(ex['text'])))
    print(f"    Length range: {min(lengths)} - {max(lengths)} tokens")
    print(f"    Mean length: {sum(lengths)/len(lengths):.0f} tokens")

    # Check 5: Trainable parameters
    print(f"\n[5] Model parameters")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"    Trainable: {trainable:,} ({pct:.2f}%)")
    print(f"    Total: {total:,}")
    if pct < 0.01 or pct > 1.0:
        print(f"    WARNING: Trainable % seems unusual for LoRA")

    # Check 6: Effective batch size
    print(f"\n[6] Batch size")
    n_gpus = torch.cuda.device_count()
    effective_bs = (
        args.per_device_train_batch_size *
        args.gradient_accumulation_steps *
        max(n_gpus, 1)
    )
    print(f"    Per-device: {args.per_device_train_batch_size}")
    print(f"    Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"    GPUs detected: {n_gpus}")
    print(f"    Effective batch size: {effective_bs}")
    if effective_bs != 32:
        print(f"    NOTE: Paper uses batch size 32, you have {effective_bs}")

    # Check 7: Special tokens
    print(f"\n[7] Special tokens")
    print(f"    PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"    EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"    BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")

    # Check 8: Completion-only loss fields
    print(f"\n[8] Completion-only loss")
    sample = train_dataset[0]
    has_prompt = 'prompt' in sample
    has_completion = 'completion' in sample
    print(f"    'prompt' field: {'present' if has_prompt else 'MISSING!'}")
    print(f"    'completion' field: {'present' if has_completion else 'MISSING!'}")
    if has_prompt and has_completion:
        print(f"    TRL will auto-enable completion-only loss")
    else:
        print(f"    WARNING: Dataset must have 'prompt' and 'completion' fields!")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All sanity checks PASSED")
    else:
        print("Some sanity checks FAILED - review warnings above")
    print("=" * 60)

    return all_passed


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: TrainingArgs,
) -> SFTTrainer:
    """Create the SFTTrainer with completion-only loss.

    TRL 0.26+ uses completion_only_loss parameter in SFTConfig.
    It auto-detects prompt/completion fields in the dataset.
    """

    # Generate run name
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"cogex-{args.model_name}-{timestamp}"
    else:
        run_name = args.wandb_run_name

    # Create SFT config
    sft_config = SFTConfig(
        output_dir=args.output_dir,

        # Training
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning rate
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        # Memory
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,

        # Completion-only loss (TRL 0.26+)
        # Auto-detects prompt/completion fields in dataset
        completion_only_loss=True,

        # Logging and saving
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,

        # WandB
        report_to="wandb" if args.use_wandb else "none",
        run_name=run_name,

        # Misc
        remove_unused_columns=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    return trainer


def parse_args() -> TrainingArgs:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='COGEX Finetuning Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument('--model', type=str, default='Llama-3.1-8B',
                        help='Model name (directory under MODELS_PATH)')
    parser.add_argument('--attn-impl', type=str, default='auto',
                        choices=['auto', 'flash_attention_2', 'sdpa', 'eager'],
                        help='Attention implementation')

    # Data
    parser.add_argument('--train-file', type=str, default='data/train.json',
                        help='Training data file')
    parser.add_argument('--val-file', type=str, default='data/val.json',
                        help='Validation data file')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/cogex',
                        help='Output directory for checkpoints')

    # Training
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Per-device batch size')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max-length', type=int, default=2048,
                        help='Maximum sequence length')

    # LoRA
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout')

    # WandB
    parser.add_argument('--wandb-project', type=str, default='cogex-replication',
                        help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='WandB run name (auto-generated if not specified)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip-sanity-checks', action='store_true',
                        help='Skip pre-training sanity checks')

    cli_args = parser.parse_args()

    # Convert to TrainingArgs
    return TrainingArgs(
        model_name=cli_args.model,
        attn_implementation=cli_args.attn_impl,
        train_file=cli_args.train_file,
        val_file=cli_args.val_file,
        output_dir=cli_args.output_dir,
        num_train_epochs=cli_args.epochs,
        learning_rate=cli_args.lr,
        per_device_train_batch_size=cli_args.batch_size,
        gradient_accumulation_steps=cli_args.grad_accum,
        lora_r=cli_args.lora_r,
        lora_alpha=cli_args.lora_alpha,
        lora_dropout=cli_args.lora_dropout,
        wandb_project=cli_args.wandb_project,
        wandb_run_name=cli_args.wandb_run_name,
        use_wandb=not cli_args.no_wandb,
        seed=cli_args.seed,
    ), cli_args.skip_sanity_checks


def main():
    print("=" * 60)
    print("COGEX Finetuning")
    print("=" * 60)

    # Parse arguments
    args, skip_sanity = parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"\n[Config] Seed: {args.seed}")

    # Load environment
    models_path = load_env()
    print(f"[Config] MODELS_PATH: {models_path}")

    # Initialize WandB
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model': args.model_name,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'learning_rate': args.learning_rate,
                'epochs': args.num_train_epochs,
                'batch_size': args.per_device_train_batch_size,
                'grad_accum': args.gradient_accumulation_steps
            },
        )
        print(f"[WandB] Initialized: {args.wandb_project}")

    # Load data
    train_dataset, val_dataset = load_datasets(args)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args, models_path)

    # Apply LoRA
    model = setup_lora(model, args)

    # Create trainer
    trainer = create_trainer(
        model, tokenizer, train_dataset, val_dataset, args
    )

    # Run sanity checks
    if not skip_sanity:
        checks_passed = run_sanity_checks(
            model, tokenizer, train_dataset, val_dataset, args
        )
        if not checks_passed:
            print("\nWARNING: Some sanity checks failed. Continuing anyway...")
            print("Use --skip-sanity-checks to suppress this warning.\n")

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    print(f"\n[Save] Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save training args
    args_path = Path(args.output_dir) / "training_args.json"
    with open(args_path, 'w') as f:
        json.dump(vars(args) if hasattr(args, '__dict__') else str(args), f, indent=2, default=str)

    # Finish WandB
    if args.use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
