
python /workspace/web_search_summarize_prompt.py \
    --wsd_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence.csv \
    --output_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_summarize_prompt.csv \
    --retrieval_result_path /workspace/data/test_set_process/wsd_set_entire_google_vision_result.csv \
    --prompt_type ambig_sentence