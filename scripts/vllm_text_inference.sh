#model_checkpoint="Qwen/Qwen3-VL-30B-A3B-Thinking"
model_checkpoint="LGAI-EXAONE/EXAONE-4.5-33B"
#model_checkpoint="/workspace/model_dir/gemma3-4b-it/pos-descriminator/final_model"
#    --example_set_path /workspace/data/dataset_construction_train/invalid1_examples.csv \
#    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1

python /workspace/vllm_inference.py \
    --model_checkpoint ${model_checkpoint} \
    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_sense_ambig_sentence_text_prompt.csv \
    --output_file_path /workspace/data/test_set_process/inference/wsd_set_entire_sense_ambig_sentence_text_exaone-4d5-33b_text.csv \
    --seed 42