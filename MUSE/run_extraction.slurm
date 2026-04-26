#!/bin/bash
#SBATCH --job-name=extraction
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=extraction-%j.out
#SBATCH --error=extraction-%j.err

# Go to your project directory
cd /nfs/home/jose/unlearned_data_extraction_llm_dp/MUSE

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tofu

# Set CUDA paths
export PATH=/nfs/soft/nvidia/cuda-11.8.0/bin:$PATH
export LD_LIBRARY_PATH=/packages/nvidia/cuda-11.8.0/lib64:$LD_LIBRARY_PATH

export HF_HOME=/home/jose/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/jose/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/jose/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Sanity check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run the script
bash eval_idea_10_v2.sh
