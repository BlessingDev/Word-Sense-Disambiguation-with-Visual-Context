#model_checkpoint="google/gemma-3-27b-it"
model_checkpoint="Qwen/Qwen3-VL-4B-Instruct"
#model_checkpoint="/workspace/model_dir/gemma-3-4b-it/iwsd2/final_model"
#    --example_set_path /workspace/data/dataset_construction_train/invalid1_examples.csv \
#    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/
#    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/
#    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_sense_ambig_sentence_prompt.csv

python /workspace/vllm_inference.py \
    --model_checkpoint ${model_checkpoint} \
    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_sense_ambig_sentence_prompt.csv \
    --output_file_path /workspace/data/test_set_process/inference/wsd_set_entire_sense_ambig_sentence_sense_qwen3-vl-4b-instruct.csv \
    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
    --image_number 1 \
    --seed 42