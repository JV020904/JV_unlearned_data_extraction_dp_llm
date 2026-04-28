# define variable
port=18765
model_family="phi"
split="forget10"
minus_values=(5.0)

# Set my model paths here
model_path="/nfs/home/jose/unlearned_data_extraction_llm_dp/checkpoint_updated/MUSE/clean_dp_forget"
pretrained_path="/nfs/home/jose/unlearned_data_extraction_llm_dp/checkpoint_updated/MUSE/clean_dp_full"


for minus in "${minus_values[@]}"; do
    echo "Processing minus value: $minus"

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --master_port=$port evaluate_util.py \
        model_family=$model_family \
        batch_size=50 \
        split=$split \
        model_path=$model_path \
        +minus_value=$minus \
        +pretrained_path=$pretrained_path \
        --config-name=eval_idea.yaml
done
