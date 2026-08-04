model_checkpoint="google/gemma-3-27b-it"
#model_checkpoint="Qwen/Qwen3-VL-30B-A3B-Instruct"
#model_checkpoint="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
#model_checkpoint="LGAI-EXAONE/EXAONE-4.5-33B"
#model_checkpoint="/workspace/model_dir/Qwen3-VL-4B-Instruct/iwsd2/final_model"
#    --example_set_path /workspace/data/dataset_construction_train/invalid1_examples.csv \
#    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/
#    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/
#    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_sense_ambig_sentence_prompt.csv

python /workspace/vllm_inference.py \
    --model_checkpoint ${model_checkpoint} \
    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_sense_search_k2_mistral3_prompt.csv \
    --output_file_path /workspace/data/test_set_process/inference/wsd_set_entire_labeled_sense_search_k2_mistral3_gemma-3-27b-it.csv \
    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
    --image_number 1 \
    --seed 42