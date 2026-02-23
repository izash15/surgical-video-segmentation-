#!/bin/bash
#SBATCH --job-name=dsad_prep
#SBATCH --output=dsad_prep_%j.out
#SBATCH --error=dsad_prep_%j.err
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd ~/surgical-video-segmentation-/src/data

# sanity
/home/rmosad/.conda/envs/svs/bin/python -c "from PIL import Image; print('PIL ok in job')"

bash preprocess.sh
