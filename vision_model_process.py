

def resize_image_gemma3(image):
    """
    Resize the input image to 896x896 pixels.

    :param image: Input image as a PIL Image object
    :return: Resized and normalized image as a PyTorch tensor
    """
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize((896, 896)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = preprocess(image)
    return image_tensor

def caption_image_with_gemma3(target_word_list, image_name_list, image_path):
    """
    caption_image_with_gemma3의 Docstring
    
    :param target_word_list: 이미지에서 캡션에 포함되어야 하는 단어들의 리스트
    :param image_name_list: 이미지 파일 이름의 리스트
    :param image_path: 이미지 파일이 들어있는 디렉토리 경로
    """
    
    assert len(target_word_list) == len(image_name_list), "Target word list and image name list must have the same length."
    
    from transformers import pipeline
    import torch
    from tqdm.auto import tqdm
    
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-12b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    caption_list = list()
    for target_word, image_name in tqdm(zip(target_word_list, image_name_list), total=len(target_word_list)):
        full_image_path = f"{image_path}/{image_name}"
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": full_image_path},
                    {"type": "text", "text": f"Caption the image with a sentence including the word '{target_word}'. Do not indicate you are captioning an image. Do not mention 'The/This image', 'The/This photograph', etc. Do not highlight the given word in the sentence."}
                ]
            }
        ]
        output = pipe(text=messages, max_new_tokens=200, max_length=None)
        
        caption = output[0]["generated_text"][-1]["content"]
        caption_list.append(caption)
    
    return caption_list

def caption_one_image_with_gemma3(target_word, image_name, image_path):
    """
    caption_one_image_with_gemma3의 Docstring
    
    :param target_word: 이미지에서 캡션에 포함되어야 하는 단어
    :param image_name: 이미지 파일 이름
    :param image_path: 이미지 파일이 들어있는 디렉토리 경로
    """
    from transformers import pipeline
    import torch
    
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-12b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    full_image_path = f"{image_path}/{image_name}"
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": full_image_path},
                {"type": "text", "text": f"Caption the image with a sentence including the word '{target_word}'. Do not indicate you are captioning an image. Do not mention 'The/This image', 'The/This photograph', etc. Do not highlight the given word in the sentence."}
            ]
        }
    ]
    output = pipe(text=messages, max_new_tokens=200, max_length=None)
    
    print(output[0]["generated_text"][-1]["content"])

def answer_wsd_set_with_gemma3_image_provided(wsd_set_path, model_path="google/gemma-3-12b-it"):
    """
    answer_wsd_set_with_gemma3_image_provided의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    """
    from transformers import pipeline
    import torch
    import pandas as pd
    import json
    from tqdm.auto import tqdm
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    wsd_df = pd.read_csv(wsd_set_path)
    
    answer_list = list()
    
    for _, row in tqdm(wsd_df.iterrows(), total=len(wsd_df)):
        sense_dict = json.loads(row["senses"])
        sense_choice_string = ""
        for i, sense in enumerate(sense_dict["senses"]):
            sense_choice_string += f"{i + 1}) {sense}\n" # '1', '2', '3' 형태로 sense choice 문자열 생성
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/{}".format(row["gold_image"])},
                    {"type": "text", "text": 
                    "Context: {0}\n".format(row["generated_ambiguous_sentence"]) +
                    "Perform Word Sense Disambiguation (WSD) for a specific word."
                    "A sentence containining the word and an image describing the word's sense is given. Leverage both information to answer the question.\n"
                    "\n\nWord of interest: {0}\n".format(row["word"]) +
                    "Sense list: \n" + sense_choice_string +
                    "Only answer with the number corresponding to the correct sense (1, 2, or 3). Do not provide any explanation for your answer."
                    }
                ]
            }
        ]
        output = pipe(text=messages, max_new_tokens=200, max_length=None)
        
        answer_list.append(int(output[0]["generated_text"][-1]["content"].strip()))
        
    return answer_list

def answer_wsd_set_with_gemma3_text_only(wsd_set_path, model_path="google/gemma-3-12b-it"):
    """
    answer_wsd_set_with_gemma3_text_only의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    """
    from transformers import pipeline
    import torch
    import pandas as pd
    import json
    from tqdm.auto import tqdm
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    wsd_df = pd.read_csv(wsd_set_path)
    
    answer_list = list()
    
    for _, row in tqdm(wsd_df.iterrows(), total=len(wsd_df)):
        sense_dict = json.loads(row["senses"])
        sense_choice_string = ""
        for i, sense in enumerate(sense_dict["senses"]):
            sense_choice_string += f"{i + 1}) {sense}\n" # '1', '2', '3' 형태로 sense choice 문자열 생성
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 
                    "Context: {0}\n".format(row["generated_ambiguous_sentence"]) +
                    "Perform Word Sense Disambiguation (WSD) for a specific word."
                    "A sentence containining the word is given. Leverage the information to answer the question.\n"
                    "\n\nWord of interest: {0}\n".format(row["word"]) +
                    "Sense list: \n" + sense_choice_string +
                    "Only answer with the number corresponding to the correct sense (1, 2, or 3). Do not provide any explanation for your answer."
                    }
                ]
            }
        ]
        output = pipe(text=messages, max_new_tokens=200, max_length=None)
        
        answer_list.append(int(output[0]["generated_text"][-1]["content"].strip()))
        
    return answer_list

def answer_wsd_set_with_gemma3_broad_caption(wsd_set_path, model_path="google/gemma-3-12b-it"):
    """
    answer_wsd_set_with_gemma3_broad_caption의 Docstring
    이미지를 주는 대신에 이미지의 전체적인 캡션을 제공하는 방식으로 WSD 문제를 푸는 함수
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    """
    from transformers import pipeline
    import torch
    import pandas as pd
    import json
    from tqdm.auto import tqdm
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    wsd_df = pd.read_csv(wsd_set_path)
    
    caption_list = list()
    answer_list = list()
    
    for _, row in tqdm(wsd_df.iterrows(), total=len(wsd_df)):
        # caption 생성하기
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/{}".format(row["gold_image"])},
                    {"type": "text", "text": "Describe what is depicted in the image within a short paragraph."}
                ]
            }
        ]
        
        caption_output = pipe(text=messages, max_new_tokens=200, max_length=None)
        caption = caption_output[0]["generated_text"][-1]["content"]
        caption_list.append(caption)
        
        sense_dict = json.loads(row["senses"])
        sense_choice_string = ""
        for i, sense in enumerate(sense_dict["senses"]):
            sense_choice_string += f"{i + 1}) {sense}\n" # '1', '2', '3' 형태로 sense choice 문자열 생성
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 
                    "Image Caption: {0}\n".format(caption) +
                    "Context: {0}\n".format(row["generated_ambiguous_sentence"]) +
                    "Perform Word Sense Disambiguation (WSD) for a specific word."
                    "A sentence containining the word and a caption of an image describing the word's sense is given. Leverage both information to answer the question.\n"
                    "\n\nWord of interest: {0}\n".format(row["word"]) +
                    "Sense list: \n" + sense_choice_string +
                    "Only answer with the number corresponding to the correct sense (1, 2, or 3). Do not provide any explanation for your answer."
                    }
                ]
            }
        ]
        output = pipe(text=messages, max_new_tokens=200, max_length=None)
        
        answer_list.append(int(output[0]["generated_text"][-1]["content"].strip()))
        
    return (answer_list, caption_list)

def answer_wsd_set_with_gemma3_specific_caption(wsd_set_path, model_path="google/gemma-3-12b-it"):
    """
    answer_wsd_set_with_gemma3_specific_caption의 Docstring
    이미지를 주는 대신에 이미지의 관심 있는 부분에 대한 상세한 캡션을 제공하는 방식으로 WSD 문제를 푸는 함수
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    """
    from transformers import pipeline
    import torch
    import pandas as pd
    import json
    from tqdm.auto import tqdm
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    wsd_df = pd.read_csv(wsd_set_path)
    
    caption_list = list()
    answer_list = list()
    
    for _, row in tqdm(wsd_df.iterrows(), total=len(wsd_df)):
        # sense description 문자열 만들기
        sense_dict = json.loads(row["senses"])
        sense_choice_string = ""
        for i, sense in enumerate(sense_dict["senses"]):
            sense_choice_string += f"{i + 1}) {sense}\n" # '1', '2', '3' 형태로 sense choice 문자열 생성
        
        # caption 생성하기
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/{}".format(row["gold_image"])},
                    {"type": "text", "text": 
                    "The word of interest is '{0}'. Focus on the part of the image that is relevant to the word. Then provide a brief explanation how the word can be interpreted following given image within the sentence '{1}'.\n".format(row["word"], row["generated_ambiguous_sentence"]) +
                    "Generate the explanation without any additional text or formatting."
                    }
                ]
            }
        ]
        
        caption_output = pipe(text=messages, max_new_tokens=200, max_length=None)
        caption = caption_output[0]["generated_text"][-1]["content"]
        caption_list.append(caption)
        
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 
                    "Image Explanation: {0}\n".format(caption) +
                    "Context: {0}\n".format(row["generated_ambiguous_sentence"]) +
                    "Perform Word Sense Disambiguation (WSD) for a specific word."
                    "A sentence containining the word and a caption of an image describing the word's sense is given. Leverage both information to answer the question.\n"
                    "\n\nWord of interest: {0}\n".format(row["word"]) +
                    "Sense list: \n" + sense_choice_string +
                    "Only answer with the number corresponding to the correct sense (1, 2, or 3). Do not provide any explanation for your answer."
                    }
                ]
            }
        ]
        output = pipe(text=messages, max_new_tokens=200, max_length=None)
        
        answer_list.append(int(output[0]["generated_text"][-1]["content"].strip()))
        
    return (answer_list, caption_list)

def generate_ambiguous_sentences_with_gemma3(wsd_set_path):
    """
    generate_ambiguous_sentence_with_gemma3의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    """
    from transformers import pipeline
    from tqdm.auto import tqdm
    import pandas as pd
    import json
    import torch
    
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-27b-it",
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    wsd_df = pd.read_csv(wsd_set_path)
    
    sentence_list = list()
    batch_size = 8
    message_batch = list()
    tqdm_length = int(len(wsd_df) / batch_size)
    tqdm_bar = tqdm(total=tqdm_length, desc="Generating ambiguous sentences")
    for _, row in wsd_df.iterrows():
        target_word = row["word"]
        sense_dict = json.loads(row["senses"])
        if "noun_senses" in sense_dict:
            sense_dict["senses"] = sense_dict["noun_senses"]
        elif "verb_senses" in sense_dict:
            sense_dict["senses"] = sense_dict["verb_senses"]
        
        sense_choice_string = ""
        for i, sense in enumerate(sense_dict["senses"]):
            sense_choice_string += f"{i + 1}) {sense} " # 1, 2, 3 형태로 sense choice 문자열 생성
        interested_sense = sense_dict["senses"][int(row["gold_sense"]) - 1] # 예시에서는 첫 번째 sense가 관심있는 의미라고 가정
    
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 
                    f"The senses of '{target_word}' are \n{sense_choice_string.strip()}.\n\n"
                    f"Generate a natural ambiguous sentence using polysemous word '{target_word}' once so that the sentence can be interpreted in as much meaning as possible.\n"
                    f"The word '{target_word}' in the generated sentence should be able to be interpreted as sense {row['gold_sense']}) {interested_sense}.\n"
                    f"Do not highlight the word '{target_word}' in the sentence.\n"
                    "Plan, construct, and verify step by step how to structure an ambiguous sentence so it can be interpreted in multiple ways. And then suggest the final sentence with prefix 'Ambiguous sentence: '\n"
                    }
                ]
            }
        ]
        message_batch.append(messages)
        
        if len(message_batch) % batch_size == 0 and len(message_batch) > 0:
            outputs = pipe(text=message_batch, max_new_tokens=1024, max_length=None)
            sentence_list.extend([outputs[idx][0]["generated_text"][-1]["content"].strip() for idx in range(batch_size)])
            tqdm_bar.update(1)
            message_batch = list()
    
    wsd_df["generated_ambiguous_sentence"] = sentence_list
    return wsd_df

def verify_sense_labeling_with_gemma3(wsd_set_path):
    """
    verify_sense_labeling_with_gemma3의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    """
    from transformers import pipeline
    import torch
    import pandas as pd
    from tqdm.auto import tqdm
    import json
    
    wsd_df = pd.read_csv(wsd_set_path)
    
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-27b-it",
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    batch_size = 4
    answer_list = list()
    message_batch = list()
    tqdm_length = int(len(wsd_df) / batch_size)
    if len(wsd_df) % batch_size != 0:
        tqdm_length += 1
    tqdm_bar = tqdm(total=tqdm_length, desc="Verifying sense labeling")
    for _, row in wsd_df.iterrows():
        
        
        sense_list = json.loads(row["senses"])
        if "noun_senses" in sense_list:
            sense_list = sense_list["noun_senses"]
        elif "verb_senses" in sense_list:
            sense_list = sense_list["verb_senses"]
        
        sense_choice_string = ""
        for i, sense in enumerate(sense_list):
            sense_choice_string += f"{i + 1}) {sense} " # 1, 2, 3 형태로 sense choice 문자열 생성
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": f"/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/{row['gold_image']}"},
                    {"type": "text", "text": 
                    f"The sense of '{row['word']}' are {sense_choice_string.strip()}.\n" +
                    "Among the senses of '{0}', '{1}) {2}' is determined as the gold sense for the image.\n".format(row["word"], row["gold_sense"], sense_list[row["gold_sense"] - 1]) +
                    "First judge whether the determination of the gold sense is correct by looking at the image. If it is correct, just say correct without any other words. If it is not correct, find out the correct sense and provide the explanation."
                    }
                ]
            }
        ]
        
        message_batch.append(messages)
        
        if len(message_batch) % batch_size == 0 and len(message_batch) > 0:
            outputs = pipe(text=message_batch, max_new_tokens=1024, max_length=None)
            answer_list.extend([outputs[idx][0]["generated_text"][-1]["content"].strip() for idx in range(batch_size)])
            tqdm_bar.update(1)
            message_batch = list()
    
    if len(message_batch) > 0:
        outputs = pipe(text=message_batch, max_new_tokens=1024, max_length=None)
        answer_list.extend([outputs[idx][0]["generated_text"][-1]["content"].strip() for idx in range(len(message_batch))])
        tqdm_bar.update(1)
    
    wsd_df["sense_verification"] = answer_list
    
    return wsd_df

if __name__ == "__main__":
    from transformers import pipeline
    import torch
    
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-27b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/image.4522.jpg"},
                {"type": "text", "text": 
                "The sense of 'stamp' are "
                "1) the distinctive form in which a thing is made 2) a type or class 3) a symbol that is the result of printing or engraving 4) a small adhesive token stuck on a letter or package to indicate that that postal fees have been paid 5) something that can be used as an official medium of payment 6) a small piece of adhesive paper that is put on an object to show that a government tax has been paid 7) machine consisting of a heavy bar that moves vertically for pounding or crushing ores 8) a block or die used to imprint a mark or design \n\n"
                "Among the senses of 'stamp', '5) something that can be used as an official medium of payment' is determined as the gold sense for the image.\n"
                "First judge whether the determination of the gold sense is correct by looking at the image. If it is correct, just say correct without any other words. If it is not correct, find out the correct sense and provide the explanation."
                }
            ]
        }
    ]
    output = pipe(text=messages, max_new_tokens=1024, max_length=None)
    
    print(output[0]["generated_text"][-1]["content"])