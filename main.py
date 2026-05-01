def rearrange_trial_to_text_wsd(output_file_path):
    """
    rearrange_trial_to_text_wsd의 Docstring
    
    :param output_file_path: 재배열된 데이터를 저장할 파일 경로
    """
    import data_process
    
    data_path = '/workspace/data/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.data.v1.txt'
    gold_image_path = '/workspace/data/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.gold.v1.txt'
    
    polysemy_words_df = data_process.filter_polysemy_words(data_path)
    word_index_list = polysemy_words_df['word_index'].tolist()
    gold_image_list = data_process.get_gold_image_list(gold_image_path, word_index_list)
    
    polysemy_words_df['gold_image'] = gold_image_list
    
    polysemy_words_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"Rearranged data saved to {output_file_path}")

def sample_and_save_wsd_set(output_file_path, sample_size, check_duplicates=False,
                            data_path='/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt',
                            gold_image_path='/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt'):
    """
    sample_and_save_wsd_set의 Docstring
    
    :param output_file_path: 샘플링된 데이터를 저장할 파일 경로
    :param sample_size: 샘플링할 데이터의 크기 (-1인 경우 전체 데이터셋 사용)
    :param check_duplicates: 중복 단어를 허용할지 여부 (True인 경우 중복 단어 제거)
    """
    import data_process
    
    sampled_wsd_df = data_process.sample_data_from_train_set(data_path, sample_size, check_duplicates)
    
    word_index_list = sampled_wsd_df['word_index'].tolist()
    gold_image_list = data_process.get_gold_image_list(gold_image_path, word_index_list)
    
    sampled_wsd_df['gold_image'] = gold_image_list
    
    sampled_wsd_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"Sampled WSD set saved to {output_file_path}")

def generate_ambiguous_sentence_for_wsd_set(wsd_set_path):
    """
    generate_ambiguous_sentence_for_wsd_set의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    from vision_model_process import generate_ambiguous_sentences_with_gemma3
    
    wsd_df = generate_ambiguous_sentences_with_gemma3(wsd_set_path)
    
    wsd_df.to_csv('/workspace/data/train_set_process/wsd_set_n400_ambiguous.csv', index=False, encoding='utf-8')

def caption_trail_images(word_df_path):
    """
    caption_trail_images의 Docstring
    """
    import pandas as pd
    from vision_model_process import caption_image_with_gemma3
    
    image_base_path = '/workspace/data/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1'
    
    polysemy_words_df = pd.read_csv(word_df_path)
    
    target_word_list = polysemy_words_df['word'].tolist()
    image_name_list = polysemy_words_df['gold_image'].tolist()
    
    captions = caption_image_with_gemma3(target_word_list, image_name_list, image_base_path)
    
    polysemy_words_df['generated_caption'] = captions
    
    output_captioned_data_path = '/workspace/data/captioned_polysemy_words.csv'
    polysemy_words_df.to_csv(output_captioned_data_path, index=False, encoding='utf-8')
    print(f"Captioned data saved to {output_captioned_data_path}")

def verify_sense_labels(wsd_set_path, output_file_path):
    """
    verify_sense_labels의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 검증된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    from vision_model_process import verify_sense_labeling_with_gemma3
    
    wsd_df = verify_sense_labeling_with_gemma3(wsd_set_path)
    
    wsd_df.to_csv(output_file_path, index=False, encoding='utf-8')

def make_ambiguous_sentence_generation_prompt(wsd_set_path, output_file_path):
    """
    make_ambiguous_sentence_generation_prompt의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 프롬프트가 포함된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    from data_process import ambiguous_sentence_generation_prompts_for_wsd_set
    
    wsd_df = ambiguous_sentence_generation_prompts_for_wsd_set(wsd_set_path)
    
    wsd_df.to_csv(output_file_path, index=False, encoding='utf-8')

def answer_wsd_set_and_save(wsd_df_path, model_path, method="image", output_file_path="/workspace/data/"):
    """
    answer_wsd_set_and_save의 Docstring
    
    :param wsd_df_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param model_path: 사용할 모델의 경로
    :param method: 사용할 답변 방법 ("image", "text", "broad_caption", "specific_caption" 중 하나)
    :param output_file_path: 답변된 WSD 세트를 저장할 파일 경로
    """
    import pandas as pd
    from vision_model_process import answer_wsd_set_with_gemma3_image_provided, answer_wsd_set_with_gemma3_text_only, answer_wsd_set_with_gemma3_broad_caption, answer_wsd_set_with_gemma3_specific_caption
    
    wsd_df = pd.read_csv(wsd_df_path)
    
    if method == "image":
        answer_list = answer_wsd_set_with_gemma3_image_provided(wsd_df_path, model_path=model_path)
    elif method == "text":
        answer_list = answer_wsd_set_with_gemma3_text_only(wsd_df_path, model_path=model_path)
    elif method == "broad_caption":
        answer_list, caption_list = answer_wsd_set_with_gemma3_broad_caption(wsd_df_path, model_path=model_path)
        
        wsd_df["generated_caption"] = caption_list
    elif method == "specific_caption":
        answer_list, caption_list = answer_wsd_set_with_gemma3_specific_caption(wsd_df_path, model_path=model_path)
        
        wsd_df["generated_caption"] = caption_list
    
    wsd_df["predicted_answer"] = answer_list
    
    output_answered_data_path = output_file_path + f"{model_path.split('/')[-1]}/answered_ambiguous_sentence_wsd_set_{method}.csv"
    wsd_df.to_csv(output_answered_data_path, index=False, encoding='utf-8')
    print(f"Answered WSD set saved to {output_answered_data_path}")

def google_vision_search_for_wsd_set(wsd_df_path, output_file_path, image_dir="/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/"):
    """
    google_vision_search_for_wsd_set의 Docstring
    
    :param wsd_df_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 이미지 검색 결과가 포함된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    import os
    import json
    import pandas as pd
    from tqdm.auto import tqdm
    from image_search import perform_google_vision_search
    
    wsd_df = pd.read_csv(wsd_df_path)
    
    target_word_list = wsd_df['gold_image'].tolist()
    
    retriev_result_dict = {
        "word_index": wsd_df['word_index'].tolist(),
        "web_urls": [],
        "best_label": [],
        "entities": []
    }
    for idx in tqdm(range(len(target_word_list))):
        target_word_list[idx] = os.path.join(image_dir, target_word_list[idx])
    
        retrieved_result = perform_google_vision_search(target_word_list[idx])
        
        if retrieved_result is None:
            retriev_result_dict["web_urls"].append("None")
            retriev_result_dict["best_label"].append("No label")
            retriev_result_dict["entities"].append("None")
        else:
            retriev_result_dict["web_urls"].append(json.dumps(retrieved_result["web_urls"]))
            retriev_result_dict["best_label"].append(retrieved_result["best_label"])
            retriev_result_dict["entities"].append(json.dumps(retrieved_result["entities"]))

    retrieved_df = pd.DataFrame(retriev_result_dict)
    retrieved_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"WSD set with retrieved image result saved to {output_file_path}")

def main():
    #output_file_path = '/workspace/data/rearranged_polysemy_words.csv'
    #rearrange_trial_to_text_wsd(output_file_path)
    #sample_and_save_wsd_set("/workspace/data/set_process/wsd_set_test.csv", sample_size=-1, check_duplicates=True)
    #sample_and_save_wsd_set("/workspace/data/test_set_benchmark/wsd_set_100.csv", sample_size=100, check_duplicates=False, data_path="/workspace/data/semeval-2023-V-WSD-test/en.test.data.v1.1.txt", gold_image_path="/workspace/data/semeval-2023-V-WSD-test/en.test.gold.v1.1.txt")
    #caption_trail_images(output_file_path)
    
    model_path = "google/gemma-3-12b-it"
    #answer_wsd_set_and_save('/workspace/data/train_sample_processed/ambiguous_sentence_wsd_set.csv', model_path=model_path, method="specific_caption", output_file_path="/workspace/data/train_sample_processed/")
    
    '''verify_sense_labels(
        "/workspace/data/train_set_process/wsd_set_extra_ambiguous.csv",
        "/workspace/data/train_set_process/wsd_set_extra_ambiguous_verified.csv"
    )'''
    '''make_ambiguous_sentence_generation_prompt(
        "/workspace/data/train_set_process/wsd_set_extra_ambiguous.csv",
        "/workspace/data/train_set_process/wsd_set_extra_ambiguous_prompt.csv"
    )'''
    
    #generate_ambiguous_sentence_for_wsd_set("/workspace/data/train_set_process/wsd_set_n400.csv")
    google_vision_search_for_wsd_set("/workspace/data/test_set_process/wsd_set_entire.csv", "/workspace/data/test_set_process/wsd_set_entire_google_vision_result.csv", image_dir="/workspace/data/semeval-2023-V-WSD-test/test_images/")
    
if __name__ == "__main__":
    main()