#!/bin/bash
#SBATCH --job-name=cogex-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=$USER@princeton.edu
#SBATCH --mail-type=ALL

module purge
module load cudatoolkit/12.4.1

cd /jukebox/graziano/jack/newcomb
source ./venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=cogex-replication

MODEL="${MODEL:-Llama-3.1-8B}"
OUTPUT_DIR="outputs/cogex-${MODEL}-$(date +%Y%m%d_%H%M%S)"

EPOCHS=5
LR=3e-4
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=2048

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

echo "========================================"
echo "COGEX Finetuning Job"
echo "========================================"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM x 2 GPUs = $((BATCH_SIZE * GRAD_ACCUM * 2))"
echo "LoRA rank: $LORA_R"
echo "========================================"
echo ""

nvidia-smi

echo ""
echo "Starting training..."
echo ""

accelerate launch \
    --num_processes=2 \
    --mixed_precision=bf16 \
    cogex/train_cogex.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --max-length $MAX_LENGTH \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "cogex-$MODEL-$(date +%Y%m%d)"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================"
