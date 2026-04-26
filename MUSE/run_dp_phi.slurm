#!/bin/bash
#SBATCH --job-name=ThesisDP
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=thesisdp-%j.out
#SBATCH --error=thesisdp-%j.err

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

# Sanity check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run your script
bash finetune_phi_all_iter_v2.sh
