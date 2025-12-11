#!/bin/bash
#SBATCH --job-name=make_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=jt1955@princeton.edu
#SBATCH --mail-type=ALL

module purge

cd /jukebox/graziano/jack/newcomb
source ./venv/bin/activate

python make_dataset.py
