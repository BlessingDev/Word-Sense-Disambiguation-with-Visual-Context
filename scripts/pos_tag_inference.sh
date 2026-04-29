
python /workspace/inference_classification_result.py \
    --model_checkpoint /workspace/model_dir/t5gemma-l-l-ul2-it/pos-tagging-classifier-augmentedx2/final_model \
    --test_file /workspace/data/dataset_construction_train/pos_tagging_test_prompt2.csv \
    --prediction_output_dir /workspace/data/dataset_construction_train/inference/pos_tagging_prompt2_t5gemma_l_l_ul2_it_augmentedx2_test_results \
    --batch_size 64 \
    --decision_threshold 0.5