python /workspace/calculate_rlhf_score.py \
    --model_checkpoint "OpenAssistant/reward-model-deberta-v3-large-v2" \
    --input_csv /workspace/data/test_set_process/inference/wsd_set_100_ambiguous_sentence2_gpt-5.4-mini.csv \
    --output_csv /workspace/data/test_set_process/inference/wsd_set_100_ambiguous_sentence2_gpt-5.4-mini_scored.csv \
    --text_column generated_sentence \
    --batch_size 64 \
    --cpu