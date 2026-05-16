import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    EarlyStoppingCallback,
)
from trl import (
    SFTTrainer,
    SFTConfig
)
import argparse
import json
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 1. SETUP: LOAD AND PARSE THE DATA
# ------------------------------------
# We'll use Flan-T5, which is excellent for instruction-based tasks.
MODEL_CHECKPOINT = "google/gemma-3-4b-it"
TRAIN_DATASET_PATH = "/workspace/data/dataset_construction_train/train_set_pos.csv"
VAL_DATASET_PATH = "/workspace/data/dataset_construction_train/val_set_pos.csv"

MAX_INPUT_LENGTH = 2048
MAX_OUTPUT_LENGTH = 256

PROMPT_TEMPLATE = """Text Context: {context}
Ambiguous Word: {word}
---
Sense List:
{sense_list}
---
You are an expert linguistic annotator. An ambiguous word found within the context of a given text, and a list of potential senses for that word are provided. Your task is to determine the correct sense by selecting the one that most directly aligns with the context and its background event.
---
Decision Logic & Rules
Follow these rules in order of priority to make your decision:

First, take a look at the 'Text Context' and the image. Leverage both context to analyze any significant background context is provided in the image. If such information exists, prioritize the sense that best fits this background information, even if it is not the most direct match for the visual content.

Directness of Meaning: If no additional background information can be obtained from the web search results, choose the sense that provides the most direct and specific fit for the visual and linguistic context. Even if a perfect match does not exist, select the sense that is the closest indirect match.
---
Provide your final decision as the format '[sense number]' and end your generation. In here, [sense number] is the index of your chosen sense from the provided list.
"""

def main(args):
    #model_name = args.model_checkpoint.split("/")[-1]

    # Create a Hugging Face Dataset
    train_dataset = Dataset.from_csv(args.train_file)
    val_dataset = Dataset.from_csv(args.validation_file)

    # Load tokenizer and model
    processor = AutoProcessor.from_pretrained(args.model_checkpoint)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    if "Qwen" in args.model_checkpoint:
        from transformers import Qwen3VLForConditionalGeneration
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_checkpoint, 
            device_map="auto",
            #attn_implementation=args.attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint, 
            device_map="auto",
        )
    

    # 2. PREPROCESS THE DATA
    # ------------------------------------
    # We frame the task with a prefix to guide the model.
    def preprocess_function(sample):
        assistant_text = str(int(sample["gold_sense"]))
        
        sense_list = json.loads(sample["senses"]).get(sample["gold_pos"], [])
        sense_str = "\n".join([f" {idx + 1}. {sense}" for idx, sense in enumerate(sense_list)])
        prompt = PROMPT_TEMPLATE.format(
            context=sample["word_phrase"],
            word=sample["word"],
            sense_list=sense_str
        )
        
        # Prepare inputs with the prefix
        return {
          "prompt": [
              {"role": "user", "content": [
                  {"type": "image", "url": os.path.join(args.image_dir, sample["gold_image"])},
                  {"type": "text", "text": prompt}
                ]}
          ],
          "completion": [
              {"role": "assistant", "content": assistant_text}
          ]
        }

    # Apply the preprocessing to our datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function)
    tokenized_val_dataset = val_dataset.map(preprocess_function)


    # 3. FINE-TUNE THE MODEL
    # ------------------------------------
    # Load the pre-trained causal LM model

    bf16_precision = torch.cuda.is_available() and model.dtype == torch.float32

    # Define training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        do_train=True,
        do_eval=True,
        packing=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 2},
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps, 
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.train_epochs, # Increase epochs for small datasets
        gradient_checkpointing=True,
        metric_for_best_model="eval_loss",
        bf16=bf16_precision, # Use mixed precision if a GPU is available
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )

    # Create the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        processing_class=processor.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.processing_class = processor
    # Start training! 🚀
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=TRAIN_DATASET_PATH)
    parser.add_argument("--validation_file", type=str, default=VAL_DATASET_PATH)
    parser.add_argument("--image_dir", type=str, default="/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1")
    parser.add_argument("--model_checkpoint", type=str, default=MODEL_CHECKPOINT)
    
    parser.add_argument(
        "--output_dir", type=str, required=True
    )
    parser.add_argument(
        "--train_epochs", type=int, default=10
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500
    )
    parser.add_argument(
        "--logging_steps", type=int, default=200
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Attention implementation to use (eager, flash_attention_2, etc.)"
    )
    
    args = parser.parse_args()
    '''args = parser.parse_args(
        [
            "--model_checkpoint", "google/gemma-3-4b-it",
            "--output_dir", "/workspace/model_dir/test",
            "--train_file", "/workspace/data/dataset_construction_train/train_set_pos.csv",
            "--validation_file", "/workspace/data/dataset_construction_train/val_set_pos.csv",
            "--batch_size", "2",
            "--gradient_accumulation_steps", "2",
            "--train_epochs", "1",
        ]
    )'''
    
    print(args)
    
    main(args)