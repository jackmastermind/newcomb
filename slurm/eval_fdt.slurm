#!/bin/bash
#SBATCH --job-name=eval-fdt
#SBATCH --array=0-51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=eval-fdt-%A_%a.out
#SBATCH --mail-user=jt1955@princeton.edu
#SBATCH --mail-type=ALL

module purge
module load cudatoolkit/12.4.1

cd /jukebox/graziano/jack/newcomb
source ./venv/bin/activate

python eval_fdt.py \
    --model "${MODEL:-Llama-3.1-8B}" \
    --model-type "${MODEL_TYPE:?Set MODEL_TYPE to 'cogex' or 'chat'}" \
    --dataset "${DATASET:-data/fdt.json}" \
    --output-dir "${OUTPUT_DIR:-data/tmp}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-10}" \
    --worker-id "$SLURM_ARRAY_TASK_ID" \
    --num-workers "$SLURM_ARRAY_TASK_COUNT" \
    --single-gpu \
    ${LORA_PATH:+--lora-path "$LORA_PATH"}

EXIT_CODE=$?
echo "Job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
