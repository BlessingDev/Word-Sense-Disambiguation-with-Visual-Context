import os
import random
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.multimodal.image import convert_image_mode
from vllm.lora.request import LoRARequest

from PIL import Image
import pandas as pd
import numpy as np
import argparse
from vllm.utils.argparse_utils import FlexibleArgumentParser

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

# Gemma 3
'''def run_gemma3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "google/gemma-3-12b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True}
    )

    prompts = [("<bos><start_of_turn>user\n"
                f"<start_of_image>{question}<end_of_turn>\n"
                "<start_of_turn>model\n") for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = ImageAsset("cherry_blossom") \
            .pil_image.convert("RGB")
        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]

        return {
            "data": image,
            "questions": img_questions,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)'''

def apply_image_repeat(image_repeat_prob, num_prompts, data,
                       prompts: list[str], modality):
    """Repeats images with provided probability of "image_repeat_prob". 
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert (image_repeat_prob <= 1.0 and image_repeat_prob >= 0)
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        inputs.append({
            "prompt": prompts[i % len(prompts)],
            "multi_modal_data": {
                modality: cur_image
            }
        })

    return inputs

def is_truncated(img_path):
    try:
        with Image.open(img_path) as img:
            img.load()  # This forced loading catches the truncation
        return False
    except OSError:
        return True

def inference_image_zeroshot_gemma3(args):
    data_df = pd.read_csv(args.inference_set_path)

    engine_args = EngineArgs(
        model=args.model_checkpoint,
        max_model_len=8192,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )
    default_limits = {"image": 0, "video": 0, "audio": 0, "vision_chunk": 0}
    engine_args.limit_mm_per_prompt = default_limits | dict(
        engine_args.limit_mm_per_prompt or {}
    )
    engine_args.seed = args.seed
    engine_args.tensor_parallel_size = 4
    
    #llm = LLM(**engine_args_dict)
    llm = LLM.from_engine_args(engine_args)
    
    # split the set into executable size (for example, 500 samples per run) and run inference on each split
    row_per_run = 200
    data_splits = [data_df[i:i + row_per_run].copy() for i in range(0, data_df.shape[0], row_per_run)]
    
    answers = []
    valid_indices = []
    for split_idx, data_split in enumerate(data_splits):
        inputs = list()
        for idx, row in data_split.iterrows():
            prompts = ("<bos><start_of_turn>user\n"
            f"<start_of_image><image_soft_token><end_of_image>{row['prompt']}<end_of_turn>\n"
            "<start_of_turn>model\n")
            #prompts = prompts.replace("A: ", "")
            
            img_path = os.path.join(args.image_dir, row["gold_image"])
            
            turncated = is_truncated(img_path)
            if not turncated:
                image_file = Image.open(img_path)
                inputs.append({
                    "prompt": prompts,
                    "multi_modal_data": {
                        "image": convert_image_mode(image_file, "RGB")
                    }
                })
                valid_indices.append(idx)
            else:
                print(f"Image {img_path} is truncated. Skipping this sample.")
        
        # Greedy Decoding
        sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=2048,
                                        stop_token_ids=None)
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    
        for answer_idx, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            answers.append(generated_text)
        
        print("Completed inference for split {}/{}.".format(split_idx + 1, len(data_splits)))
        print("Answer Example: {}".format(answers[-1]))

    
    data_df.loc[valid_indices, "generated_text"] = answers
    data_df.to_csv(args.output_file_path, index=False)

def inference_text_gemma3(args):
    data_df = pd.read_csv(args.inference_set_path)

    engine_args = EngineArgs(
        model=args.model_checkpoint,
        max_model_len=8192,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
    )
    engine_args.seed = args.seed
    engine_args.tensor_parallel_size = 4
    
    #llm = LLM(**engine_args_dict)
    llm = LLM.from_engine_args(engine_args)
    
    # split the set into executable size (for example, 500 samples per run) and run inference on each split
    row_per_run = 200
    data_splits = [data_df[i:i + row_per_run].copy() for i in range(0, data_df.shape[0], row_per_run)]
    
    answers = []
    valid_indices = []
    for split_idx, data_split in enumerate(data_splits):
        inputs = list()
        for idx, row in data_split.iterrows():
            prompts = ("<bos><start_of_turn>user\n"
            f"{row['prompt']}<end_of_turn>\n"
            "<start_of_turn>model\n")
            
            
            inputs.append({
                "prompt": prompts
            })
            valid_indices.append(idx)
        
        # Greedy Decoding
        sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=2048,
                                        stop_token_ids=None)
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    
        for answer_idx, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            answers.append(generated_text)
        
        print("Completed inference for split {}/{}.".format(split_idx + 1, len(data_splits)))
        print("Answer Example: {}".format(answers[-1]))

    
    data_df.loc[valid_indices, "generated_text"] = answers
    data_df.to_csv(args.output_file_path, index=False)

def inference_image_zeroshot_qwen3(args):
    # This function is for zero-shot inference on Qwen-3, which does not require image preprocessing and can directly take image paths as input.
    # The implementation would be similar to inference_image_dataset_gemma3, but the prompts and multi-modal data format would be adjusted according to Qwen-3's requirements.
    data_df = pd.read_csv(args.inference_set_path)
    
    #model_name = "Qwen/Qwen3-VL-8B-Instruct"

    mm_limit = {"image": 1}
    engine_args = EngineArgs(
        model=args.model_checkpoint,
        max_model_len=8192,
        max_num_seqs=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt=mm_limit,
    )
    default_limits = {"image": 0, "video": 0, "audio": 0, "vision_chunk": 0}
    engine_args.limit_mm_per_prompt = default_limits | dict(
        engine_args.limit_mm_per_prompt or {}
    )
    engine_args.seed = args.seed
    engine_args.tensor_parallel_size = 4
    
    #llm = LLM(**engine_args_dict)
    llm = LLM.from_engine_args(engine_args)
    
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    
    row_per_run = 200
    data_splits = [data_df[i:i + row_per_run].copy() for i in range(0, data_df.shape[0], row_per_run)]
    
    answers = []
    valid_indices = []
    for split_idx, data_split in enumerate(data_splits):
        inputs = list()
        for idx, row in data_split.iterrows():
            prompts = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{image_placeholder}"
            f"{row['prompt']}<|im_end|>\n"
            "<|im_start|>assistant\n")
            
            img_path = os.path.join(args.image_dir, row["gold_image"])
            
            turncated = is_truncated(img_path)
            if not turncated:
                image_file = Image.open(img_path)
                inputs.append({
                    "prompt": prompts,
                    "multi_modal_data": {
                        "image": convert_image_mode(image_file, "RGB")
                    }
                })
                valid_indices.append(idx)
            else:
                print(f"Image {img_path} is truncated. Skipping this sample.")
        
        # Greedy Decoding
        sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=2048,
                                        stop_token_ids=None)
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    
        for answer_idx, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            answers.append(generated_text)
        
        print("Completed inference for split {}/{}.".format(split_idx + 1, len(data_splits)))
        print("Answer Example: {}".format(answers[-1]))

    
    data_df.loc[valid_indices, "generated_text"] = answers
    data_df.to_csv(args.output_file_path, index=False)

def inference_zeroshot_qwen3(args):
    # This function is for zero-shot inference on Qwen-3, which does not require image preprocessing and can directly take image paths as input.
    # The implementation would be similar to inference_image_dataset_gemma3, but the prompts and multi-modal data format would be adjusted according to Qwen-3's requirements.
    data_df = pd.read_csv(args.inference_set_path)
    
    #model_name = "Qwen/Qwen3-VL-8B-Instruct"

    engine_args = EngineArgs(
        model=args.model_checkpoint,
        max_model_len=8192,
        max_num_seqs=2,
    )
    engine_args.seed = args.seed
    engine_args.tensor_parallel_size = 4
    
    #llm = LLM(**engine_args_dict)
    llm = LLM.from_engine_args(engine_args)
    
    row_per_run = 200
    data_splits = [data_df[i:i + row_per_run].copy() for i in range(0, data_df.shape[0], row_per_run)]
    
    answers = []
    valid_indices = []
    for split_idx, data_split in enumerate(data_splits):
        inputs = list()
        for idx, row in data_split.iterrows():
            prompts = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{row['prompt']}<|im_end|>\n"
            "<|im_start|>assistant\n")
            
            inputs.append({
                "prompt": prompts
            })
            valid_indices.append(idx)
        
        # Greedy Decoding
        sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=2048,
                                        stop_token_ids=None)
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    
        for answer_idx, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            answers.append(generated_text)
        
        print("Completed inference for split {}/{}.".format(split_idx + 1, len(data_splits)))
        print("Answer Example: {}".format(answers[-1]))

    
    data_df.loc[valid_indices, "generated_text"] = answers
    data_df.to_csv(args.output_file_path, index=False)

def inference_image_zeroshot_exaone4d5(args):
    # This function is for zero-shot inference on Exaone-4.5, which does not require image preprocessing and can directly take image paths as input.
    # The implementation would be similar to inference_image_dataset_gemma3, but the prompts and multi-modal data format would be adjusted according to Exaone-4.5's requirements.
    data_df = pd.read_csv(args.inference_set_path)
    
    #model_name = "Qwen/Qwen3-VL-8B-Instruct"

    mm_limit = {"image": 1}
    engine_args = EngineArgs(
        model=args.model_checkpoint,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt=mm_limit,
    )
    default_limits = {"image": 0, "video": 0, "audio": 0, "vision_chunk": 0}
    engine_args.limit_mm_per_prompt = default_limits | dict(
        engine_args.limit_mm_per_prompt or {}
    )
    engine_args.seed = args.seed
    engine_args.tensor_parallel_size = 4
    
    llm = LLM.from_engine_args(engine_args)
    
    image_placeholder = "<vision><|image_pad|></vision>"
    
    row_per_run = 200
    data_splits = [data_df[i:i + row_per_run].copy() for i in range(0, data_df.shape[0], row_per_run)]
    
    answers = []
    valid_indices = []
    for split_idx, data_split in enumerate(data_splits):
        inputs = list()
        for idx, row in data_split.iterrows():
            prompts = ("<|system|>\nYou are a helpful assistant.<|endofturn|>\n"
            f"<|user|>\n{image_placeholder}"
            f"{row['prompt']}<|endofturn|>\n"
            "<|assistant|>\n")
            
            img_path = os.path.join(args.image_dir, row["gold_image"])
            
            turncated = is_truncated(img_path)
            if not turncated:
                image_file = Image.open(img_path)
                inputs.append({
                    "prompt": prompts,
                    "multi_modal_data": {
                        "image": convert_image_mode(image_file, "RGB")
                    },
                    "thinking": True
                })
                valid_indices.append(idx)
            else:
                print(f"Image {img_path} is truncated. Skipping this sample.")
        
        # Greedy Decoding
        sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=2048,
                                        stop_token_ids=None)
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    
        for answer_idx, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            answers.append(generated_text)
        
        print("Completed inference for split {}/{}.".format(split_idx + 1, len(data_splits)))
        print("Answer Example: {}".format(answers[-1]))

    
    data_df.loc[valid_indices, "generated_text"] = answers
    data_df.to_csv(args.output_file_path, index=False)

def inference_zeroshot_exaone4d5(args):
    # This function is for zero-shot inference on Exaone-4.5, which does not require image preprocessing and can directly take image paths as input.
    # The implementation would be similar to inference_image_dataset_gemma3, but the prompts and multi-modal data format would be adjusted according to Exaone-4.5's requirements.
    data_df = pd.read_csv(args.inference_set_path)
    
    #model_name = "Qwen/Qwen3-VL-8B-Instruct"

    engine_args = EngineArgs(
        model=args.model_checkpoint,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=2,
    )
    engine_args.seed = args.seed
    engine_args.tensor_parallel_size = 4
    
    #llm = LLM(**engine_args_dict)
    llm = LLM.from_engine_args(engine_args)
    
    row_per_run = 200
    data_splits = [data_df[i:i + row_per_run].copy() for i in range(0, data_df.shape[0], row_per_run)]
    
    answers = []
    valid_indices = []
    for split_idx, data_split in enumerate(data_splits):
        inputs = list()
        for idx, row in data_split.iterrows():
            prompts = ("<|system|>\nYou are a helpful assistant.<|endofturn|>\n"
            f"<|user|>\n"
            f"{row['prompt']}<|endofturn|>\n"
            "<|assistant|>\n")
            
            inputs.append({
                "prompt": prompts,
                "thinking": True
            })
            valid_indices.append(idx)
        
        # Greedy Decoding
        sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=2048,
                                        stop_token_ids=None)
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    
        for answer_idx, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            answers.append(generated_text)
        
        print("Completed inference for split {}/{}.".format(split_idx + 1, len(data_splits)))
        print("Answer Example: {}".format(answers[-1]))

    
    data_df.loc[valid_indices, "generated_text"] = answers
    data_df.to_csv(args.output_file_path, index=False)

def inference_fewshot_dataset_gemma3(args):
    data_df = pd.read_csv(args.inference_set_path)

    engine_args = EngineArgs(
        model=args.model_checkpoint,
        max_model_len=8192,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": args.image_number},
    )
    default_limits = {"image": 0, "video": 0, "audio": 0, "vision_chunk": 0}
    engine_args.limit_mm_per_prompt = default_limits | dict(
        engine_args.limit_mm_per_prompt or {}
    )
    engine_args.seed = args.seed
    engine_args.tensor_parallel_size = 4
    
    example_df = pd.read_csv(args.example_set_path)
    example_paths = example_df["gold_image"].tolist()
    example_images = [convert_image_mode(Image.open(os.path.join(args.image_dir, path)), "RGB") for path in example_paths]

    inputs = list()
    for idx, row in data_df.iterrows():
        prompts = ("<bos><start_of_turn>user\n"
        f"{row['prompt']}<end_of_turn>\n"
            "<start_of_turn>model\n")
        
        img_paths = os.path.join(args.image_dir, row["gold_image"])
        
        images = example_images + [convert_image_mode(Image.open(img_paths), "RGB")]
        
        inputs.append({
            "prompt": prompts,
            "multi_modal_data": {
                "image": images
            }
        })
    
    engine_args_dict = asdict(engine_args)
    #llm = LLM(**engine_args_dict)
    llm = LLM.from_engine_args(engine_args)
    
    # Greedy Decoding
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=2048,
                                     stop_token_ids=None)
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    
    answers = []
    for o in outputs:
        generated_text = o.outputs[0].text
        answers.append(generated_text)
        #print(generated_text)

    
    data_df["generated_text"] = answers
    data_df.to_csv(args.output_file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument("--model_checkpoint", type=str, default="google/gemma-3-12b-it",
                        help="The name of the model to use for inference.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--inference_set_path", type=str,
                        help="Path to the CSV file containing the inference set.")
    parser.add_argument("--example_set_path", type=str, default=None,
                        help="Path to the CSV file containing the example set for few-shot prompting. (Optional)")
    parser.add_argument("--output_file_path", type=str,
                        help="Path to the CSV file where the inference results will be saved.")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to the directory containing the images.")
    parser.add_argument("--image_number", type=int, default=1,
                        help="Number of images to use for inference.")
    
    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "google/gemma-3-12b-it",
        "--inference_set_path", "/workspace/data/dataset_construction_train/train_set_caption_prompt1.csv",
        "--output_file_path", "/workspace/data/dataset_construction_train/inference/caption_prompt1_results_gemma3-12b.csv",
        "--image_dir", "/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1",
        "--image_number", "1",
        "--seed", "42"
    ])'''
    
    if args.image_dir is None:
        if "gemma-3" in args.model_checkpoint.lower():
            inference_text_gemma3(args)
        elif "qwen3" in args.model_checkpoint.lower():
            inference_zeroshot_qwen3(args)
        elif "exaone-4.5" in args.model_checkpoint.lower():
            inference_zeroshot_exaone4d5(args)
    else:
        if args.example_set_path is None:
            if "gemma-3" in args.model_checkpoint.lower():
                inference_image_zeroshot_gemma3(args)
            elif "qwen3" in args.model_checkpoint.lower():
                inference_image_zeroshot_qwen3(args)
            elif "exaone-4.5" in args.model_checkpoint.lower():
                inference_image_zeroshot_exaone4d5(args)
        else:
            inference_fewshot_dataset_gemma3(args)