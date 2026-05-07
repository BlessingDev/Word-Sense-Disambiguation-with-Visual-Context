model_checkpoint="google/gemma-3-27b-it"
#model_checkpoint="/workspace/model_dir/gemma3-4b-it/pos-descriminator/final_model"
#    --example_set_path /workspace/data/dataset_construction_train/invalid1_examples.csv \
#    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1

python /workspace/vllm_inference.py \
    --model_checkpoint ${model_checkpoint} \
    --inference_set_path /workspace/data/test_set_process/wsd_set_100_sense_ambig_sentence_prompt_text_gpt-5d4-mini.csv \
    --output_file_path /workspace/data/test_set_process/inference/wsd_set_100_sense_ambig_sentence_text_gpt-5d4-mini_gemma-3-27b_text.csv \
    --image_number 1 \
    --seed 42