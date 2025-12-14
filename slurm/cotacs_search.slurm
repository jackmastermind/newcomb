#!/bin/bash
#SBATCH --job-name=cotacs-search
#SBATCH --array=0-98
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=cotacs-search-%A_%a.out
#SBATCH --mail-user=jt1955@princeton.edu
#SBATCH --mail-type=ALL

module purge
module load cudatoolkit/12.4.1

cd /jukebox/graziano/jack/newcomb
source ./venv/bin/activate

python -m cotacs.search \
    --model "${MODEL:-Llama-3.1-8B}" \
    --theory "${THEORY:?Set THEORY to CDT, FDT, or EDT}" \
    --dataset "${DATASET:-data/fdt.json}" \
    --splits "${SPLITS:-data/cotacs/splits.json}" \
    --output-dir "${OUTPUT_DIR:-data/cotacs}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --worker-id "$SLURM_ARRAY_TASK_ID" \
    --num-workers "$SLURM_ARRAY_TASK_COUNT" \
    ${LORA_PATH:+--lora-path "$LORA_PATH"}

EXIT_CODE=$?
echo "Job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
