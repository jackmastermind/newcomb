#!/bin/bash
#SBATCH --job-name=jupyter-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=$USER@princeton.edu
#SBATCH --mail-type=ALL

module purge
module load cudatoolkit/12.4.1

cd /jukebox/graziano/jack/newcomb
source ./venv/bin/activate

jupyter lab --ip=0.0.0.0 --no-browser --allow-root