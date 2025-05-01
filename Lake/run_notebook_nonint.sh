#!/bin/bash
#SBATCH --job-name=finalproj         # Job name (avoid .py here)
#SBATCH --output=../logs/finalproj_v100%j.log   # Output log (%j = job ID)
#SBATCH --partition=a100_short        # Partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16            # CPU cores
#SBATCH --mem=128G                    # Memory
#SBATCH --time=1-00:00:00             # Time limit
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=spencer.jacobs-skolik@nyulangone.org

# Activate conda environment
source ~/.bashrc                  # Ensure conda is available
conda activate dl4med_25

# Enable CUDA DSA
export TORCH_USE_CUDA_DSA=1

# Run notebook
jupyter nbconvert --to script DualFreqSleepStager.ipynb --output DualFreqSleepStager
jupyter nbconvert --to notebook --execute --allow-errors DualFreqSleepStager.ipynb --output DualFreqSleepStager_executed.ipynb

# convert executed notebook to pdf
#jupyter nbconvert --to pdf hw3_executed.ipynb --output-dir=../pdfs --PDFExporter.latex_command="['pdflatex', '{filename}', '-interaction=nonstopmode', '-output-directory={tempdir}']" --PDFExporter.verbose=True