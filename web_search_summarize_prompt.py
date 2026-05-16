import trafilatura
import requests
import json
import pandas as pd
from tqdm.auto import tqdm

def extract_clean_text(url):
    try:
        response = requests.head(url, timeout=10)
    except:
        return None

    if response.status_code == 200:
        downloaded = trafilatura.fetch_url(url)
        # 본문만 추출 (광고, 네비게이션 자동 제거)
        result = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        return result
    return None

prompt_phrase_template="""Ambiguous word: {word}
Context Phrase: {context}
Entites in Image: 
{entities}
---
You are a linguistic expert. Given 'Ambiguous Word', 'Context Phrase', and 'Entities in Image', your task is to extract and summarize any additional and helpful information from the given 'Searched Web Content' that can help explain the context of the image in relation to the 'Ambiguous Word'. Do not try to describe the image itself if there is no relevant information in the 'Searched Web Content' that can be helpful for understanding the 'Ambiguous Word'.
---
Searched Web Content:
- Title: {web_title}
- {text_context}
---
First, Refer to the 'Entities in Image' section to understand the content of the image. Then, read the 'Searched Web Content' section carefully and judge whether it is related to both '{word}' and '{context}'. If there is some helpful information, generate a word 'Relevant', otherwise generate a word 'Not Relevant'. If the first line is 'Relevant', generate a summary of the helpful information in the 'Searched Web Content' that can explain the context of the image in relation to the 'Ambiguous Word'. If the first line is 'Not Relevant', do not generate any summary and end your generation.
"""

prompt_sentence_template="""Ambiguous word: {word}
Context Sentence: {context}
Entites in Image: 
{entities}
---
You are a linguistic expert. Given 'Ambiguous Word', 'Context Sentence', and 'Entities in Image', your task is to extract and summarize any additional and helpful information from the given 'Searched Web Content' that can help explain the context of the image in relation to the 'Ambiguous Word'. Do not try to describe the image itself if there is no relevant information in the 'Searched Web Content' that can be helpful for understanding the 'Ambiguous Word'.
---
Searched Web Content:
- Title: {web_title}
- {text_context}
---
First, Refer to the 'Entities in Image' section to understand the content of the image. Then, read the 'Searched Web Content' section carefully and judge whether it is related to the usage of '{word}' in '{context}'. If there is some helpful information, generate a word 'Relevant', otherwise generate a word 'Not Relevant'. If the first line is 'Relevant', generate a summary of the helpful information in the 'Searched Web Content' that can explain the context of the image in relation to the 'Ambiguous Word'. If the first line is 'Not Relevant', do not generate any summary and end your generation.
"""

def main(args):
    wsd_test_df = pd.read_csv(args.wsd_set_path)

    # k =3
    # 요약은 한 번의 LLM 호출로 하나의 인터넷 문서만 처리하도록 구조 적용
    retrieval_df = pd.read_csv(args.retrieval_result_path)

    df_dict = {
        "word_index": [],
        "word": [],
        "word_phrase": [],
        "senses": [],
        "gold_image": [],
        "prompt": []
    }
    if args.prompt_type == "phrase":
        prompt_template = prompt_phrase_template
    elif args.prompt_type == "sentence":
        prompt_template = prompt_sentence_template
    
    for index, row in tqdm(wsd_test_df.iterrows(), total=len(wsd_test_df)):
        word_index = row["word_index"]
        retrieved_rows = retrieval_df[retrieval_df["word_index"] == word_index]
        retrieved_urls = json.loads(retrieved_rows["web_urls"].iloc[0])
        retrieved_entities = json.loads(retrieved_rows["entities"].iloc[0])
        if len(retrieved_entities) > 0:
            valid_entites = [retrieved_entities[0]["description"]]
            for entity in retrieved_entities[1:]:
                if entity["score"] >= 1.0:
                    valid_entites.append(entity["description"])
            valid_entites = "\n".join([f"- {entity}" for entity in valid_entites])
        else:
            valid_entites = ""
        
        retrieved_urls_num = 0
        url_idx = 0
        none_page_titles = list()
        while url_idx < len(retrieved_urls) and retrieved_urls_num < 3:
            cur_url = retrieved_urls[url_idx]["url"]
            cur_title = retrieved_urls[url_idx]["title"]
            if "www.youtube.com" in cur_url:
                web_content = None
            else:
                web_content = extract_clean_text(cur_url)
            
            if web_content is not None:
                web_content_split = web_content.split('\n')
                    
                web_content_list = list()
                cur_content = ""
                content_threshold = 4096
                for line in web_content_split:
                    # 현재 line이 threshold를 넘는지 확인
                    if len(line.strip()) < content_threshold:
                        next_content = cur_content + "\n" + line.strip()
                        if len(next_content) > content_threshold:
                            web_content_list.append(cur_content)
                            cur_content = line.strip()
                        else:
                            cur_content = next_content

                if len(cur_content) > 0:
                    web_content_list.append(cur_content)
                    
                for content in web_content_list:
                    df_dict["word_index"].append(word_index)
                    df_dict["word"].append(row["word"])
                    df_dict["word_phrase"].append(row["word_phrase"])
                    df_dict["senses"].append(row["senses"])
                    df_dict["gold_image"].append(row["gold_image"])
                    df_dict["prompt"].append(prompt_template.format(word=row["word"], entities=valid_entites, context=row["word_phrase"], web_title=cur_title, text_context=content))
                retrieved_urls_num += 1
            else:
                none_page_titles.append(cur_title)
            url_idx += 1
        
        # 마지막으로 3개 추가하는 동안 등장한 None 페이지 제목들을 한꺼번에 요약하도록 프롬프트 생성
        if len(none_page_titles) > 0:
            none_page_titles_str = "\n".join([f"- {title}" for title in none_page_titles])
            df_dict["word_index"].append(word_index)
            df_dict["word"].append(row["word"])
            df_dict["word_phrase"].append(row["word_phrase"])
            df_dict["senses"].append(row["senses"])
            df_dict["gold_image"].append(row["gold_image"])
            df_dict["prompt"].append(prompt_template.format(word=row["word"], entities=valid_entites, context=row["word_phrase"], web_title="etc.", text_context=none_page_titles_str))
        
    inference_df = pd.DataFrame(df_dict)
    inference_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsd_set_path", type=str, default="/workspace/data/test_set_process/wsd_set_entire.csv")
    parser.add_argument("--retrieval_result_path", type=str, default="/workspace/data/test_set_process/wsd_set_entire_google_vision_result.csv")
    parser.add_argument("--output_path", type=str, default="/workspace/data/test_set_process/wsd_set_entire_summarize_prompt.csv")
    parser.add_argument("--prompt_type", type=str, default="phrase", choices=["phrase", "sentence"], help="Whether to use the original word phrase or the generated ambiguous sentence as context in the prompt")
    args = parser.parse_args()
    
    main(args)