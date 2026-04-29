import pandas as pd
import json

# nltk.download('wordnet')

# semeval-2023 task 1 vwsd
# rearrage the data
# read the data and collect the word of interest
# identify whether the word has multiple meanings
def filter_polysemy_words(data_path):
    """
    filter_polysemy_words의 Docstring
    
    :param data_path: 데이터 파일의 경로
    :return: DataFrame 형태의 단어 목록
    """
    from nltk.corpus import wordnet as wn
    with open(data_path, 'r') as f:
        data = f.readlines()
    index_list = []
    word_list = []
    sense_list = []
    for line_idx, line in enumerate(data):
        # split 기준은 tab
        words = line.strip().split('\t')
        # 관심있는 단어는 첫 번째 열에 등장
        word_interest = words[0]
        word_senses = wn.synsets(word_interest)
        if len(word_senses) > 1:
            index_list.append(line_idx)
            word_list.append(word_interest)
            
            cur_sense_list = list()
            for sense in word_senses:
                cur_sense_list.append(sense.definition())
            
            sense_str = json.dumps({'senses': cur_sense_list}, ensure_ascii=False)
            sense_list.append(sense_str)
    
    polysemy_words = pd.DataFrame({
        'word_index': index_list,
        'word': word_list,
        'senses': sense_list
    })
    print(f"Original data size: {len(data)}, Polysemy words size: {len(polysemy_words)}")
    return polysemy_words

def get_gold_image_list(gold_image_path, word_index_list):
    """
    get_gold_image_list의 Docstring
    
    :param gold_image_path: gold image txt 파일이 저장된 경로
    :param word_index_list: word index의 리스트
    """
    gold_image_list = []
    with open(gold_image_path, 'r') as f:
        data = f.readlines()
    
    
    for word_index in word_index_list:
        gold_image_list.append(data[word_index].strip())
    return gold_image_list

def sample_data_from_train_set(train_set_path, sample_size, check_duplicates=False):
    """
    sample_data_from_train_set의 Docstring
    
    :param train_set_path: 원본 train set이 저장된 txt 파일 경로
    :param sample_size: 샘플링할 데이터의 개수 (-1인 경우 전체 데이터 사용)
    :param check_duplicates: 중복 단어를 허용할지 여부 (True인 경우 중복 단어 제거)
    :return: 샘플링된 데이터가 저장된 DataFrame
    """
    from nltk.corpus import wordnet as wn
    
    with open(train_set_path, 'r') as f:
        lines = f.readlines()
    
    word_set = set()
    row_list = list()
    for line_idx, line in enumerate(lines):
        words = line.strip().split('\t')
        current_word = words[0]
        cur_senses = wn.synsets(current_word)
        
        add_word = True
        if check_duplicates:
            if current_word in word_set:
                add_word = False
            else:
                word_set.add(current_word)
        
        if add_word:
            # noun senses only
            sense_dict = dict()
            for sense in cur_senses:
                pos = sense.pos()
                if pos == 's': # adjective satellite는 adjective로 간주
                    pos = 'a'
                
                if pos in ['n', 'v']: # noun과 verb만 고려
                    sense_dict[pos] = sense_dict.get(pos, []) + [sense.definition()]
                '''if pos not in sense_dict:
                    sense_dict[pos] = list()
                sense_dict[pos].append(sense.definition())'''
            
            # focus only polysemy words
            if any(len(senses) > 1 for senses in sense_dict.values()):
                word_phrase = words[1]
                row_list.append({
                    'word_index': line_idx,
                    'word': current_word,
                    "word_phrase": word_phrase,
                    "senses": json.dumps(sense_dict, ensure_ascii=False)
                })
    
    set_df = pd.DataFrame(row_list)
    
    if sample_size > 0:
        sampled_df = set_df.sample(n=sample_size).reset_index(drop=True)
    else:
        sampled_df = set_df
    
    return sampled_df

def ambiguous_sentence_generation_prompts_for_wsd_set(wsd_set_path):
    """
    ambiguous_sentence_generation_prompts_for_wsd_set의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    
    wsd_df = pd.read_csv(wsd_set_path)
    
    for row_idx, row in wsd_df.iterrows():
        word = row['word']
        senses = json.loads(row['senses'])
        noun_senses = senses['noun_senses'] if 'noun_senses' in senses else []
        verb_senses = senses['verb_senses'] if 'verb_senses' in senses else []
        
        is_noun_polysemy = len(noun_senses) > 1 and len(verb_senses) == 0
        is_verb_polysemy = len(verb_senses) > 1 and len(noun_senses) == 0
        
        cur_senses = noun_senses if is_noun_polysemy else verb_senses if is_verb_polysemy else []
        
        pos = 'noun' if is_noun_polysemy else 'verb' if is_verb_polysemy else 'unknown'
        prompt = f"Generate an ambiguous sentence for the {pos} '{word}' that can be interpreted as:\n"
        if len(noun_senses) > 0:
            prompt += "Noun senses:\n"
            for idx, sense in enumerate(noun_senses):
                prompt += f"{idx + 1}) {sense}\n"
        if len(verb_senses) > 0:
            prompt += "Verb senses:\n"
            for idx, sense in enumerate(verb_senses):
                prompt += f"{idx + 1}) {sense}\n"
        
        prompt += f"Generate a natural ambiguous sentence using polysemous word '{word}', once so that the sentence can be interpreted in more than one meaning.\n"
        prompt += f"In the generated sentence, the {pos} '{word}' should be interpreted as '{cur_senses[int(row['gold_sense']) - 1]}'. However, the sentence should not imply any of the given definitions too strongly."
        if pos == 'verb':
            prompt += " Remember, the verb should always be considered from the view of the subject, not object.\n"
        else:
            prompt += "\n"
        prompt += "Do not highlight the word in the sentence.\n"
        prompt += "Explain how suggesting sentence can be interpreted in multiple meanings."
        
        wsd_df.at[row_idx, 'ambiguous_sentence_prompt'] = prompt
    
    return wsd_df

if __name__ == "__main__":
    from nltk.corpus import wordnet as wn
    import evaluate
    
    bleurt_cp = "/workspace/BLEURT-20/"    
    scorer = evaluate.load("/workspace/evaluate-local/metrics/bertscore/bertscore.py")
    word_interest = "angora"
    word_senses = wn.synsets(word_interest)
    
    
    for idx1, sense1 in enumerate(word_senses):
        sense1_def = sense1.definition().replace("(", " ").replace(")", " ").strip() + ' '.join(sense1.lemma_names())
        for idx2 in range(idx1 + 1, len(word_senses)):
            sense2_def = word_senses[idx2].definition().replace("(", " ").replace(")", " ").strip() + ' '.join(word_senses[idx2].lemma_names())
            score = scorer.compute(predictions=[sense2_def], references=[sense1_def], model_type="google/byt5-large")
            print(f"BERTScore between sense {idx1} and sense {idx2}: {score['f1'][0]}")