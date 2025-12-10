#!/bin/bash
#SBATCH --job-name=
#SBATCH --nodes=                 # node count
#SBATCH --ntasks=                # total number of tasks across all nodes
#SBATCH --cpus-per-gpu= 
#SBATCH --mem=                   # eg. 16G, 32G, 64G
#SBATCH --time=0:00:00           # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # :1 for 1 GPU, :2 for two.
#SBATCH --output=%x-%j.out       # Leave this and below lines the same!
#SBATCH --mail-user=jt1955@princeton.edu
#SBATCH --mail-type=ALL

# set up node environment
module purge
module load cudatoolkit/12.4.1 # if using GPUs

# initialize project
cd /jukebox/graziano/jack/newcomb
source ./venv/bin/activate

# do whatever your job does! probably running a .py file

# REMOVE ALL INLINE COMMENTS except slurm directives from the final file