#!/bin/bash
#SBATCH --job-name=make-dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=jt1955@princeton.edu
#SBATCH --mail-type=ALL

module purge

cd /jukebox/graziano/jack/newcomb
source ./venv/bin/activate

python make-dataset/make_themes.py
python make-dataset/make_schemas.py
python make-dataset/make_descriptions.py
