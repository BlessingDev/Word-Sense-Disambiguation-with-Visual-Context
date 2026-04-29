import torch
import numpy as np
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import argparse
import json
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 1. SETUP: CHOOSE MODEL AND DATASET
# ------------------------------------
# We use flan-t5-small for a runnable example. For better performance,
# consider 'google/flan-t5-base' or 'google/flan-t5-large'.


# 2. LOAD DATASET AND TOKENIZER
# ------------------------------------

# Load the tokenizer for Flan-T5
# We must use use_fast=True to get the word_ids() mapping.


# 4. SETUP THE TRAINER
# ------------------------------------
# Data collator handles dynamic padding for batches

# Function to compute metrics during evaluation

def np_softmax(a, axis=-1) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a, axis=axis, keepdims=True)
    y=exp_a / sum_exp_a
    
    return y

def main(args):
    model_name = args.model_checkpoint.split("/")[-1].lower()

    label_names = ["noun", "verb"]
    # Create id2label and label2id mappings for the model
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    # Load the model for token classification
    # THIS IS THE KEY STEP: T5ForTokenClassification uses the T5 encoder ONLY.
    # The decoder is not used.
    
    torch.manual_seed(args.seed)
    
    model = None
    if "modernbert" in model_name:
        from transformers import ModernBertForSequenceClassification
        
        # class_weight1 = [0.5, 1.0]
        # class_weight2 = [1.0, 2.0]
        # class_weight3 = [1.0, 1.2]
        
        model = ModernBertForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            reference_compile=False
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    elif "t5gemma" in model_name:
        from transformers import T5GemmaForSequenceClassification
        model = T5GemmaForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            is_encoder_decoder=False,
            classifier_dropout_rate=args.dropout_rate,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    elif "t5gemma-2-" in model_name:
        from transformers import T5Gemma2ForSequenceClassification
        model = T5Gemma2ForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            is_encoder_decoder=False,
            classifier_dropout_rate=args.dropout_rate,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    
    #tokenizer.model_max_length = 1024

    print("Data Loading...")
    train_datasets = HFDataset.from_csv(args.train_file)
    val_datasets = HFDataset.from_csv(args.validation_file)

    def preprocess_function(batch_samples):
        encoded = tokenizer(batch_samples["prompt"], padding=True, truncation=True)
        encoded["labels"] = batch_samples["label"]
        return encoded

    train_datasets = train_datasets.map(preprocess_function, batched=True)
    val_datasets = val_datasets.map(preprocess_function, batched=True)
    print("Data Loaded.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(p):
        decision_threshold = 0.5
        predictions, labels = p
        # Simple accuracy calculation for sequence classification
        # apply decision threshold to get binary predictions
        if predictions.shape[1] == 1:
            predictions = (predictions > decision_threshold).astype(int)
        else:
            probabilities = np_softmax(predictions, axis=1)
            predictions = np.argmax(predictions, axis=1)
            predictions[:] = (probabilities[:, 1] > decision_threshold).astype(int)
        #predictions = np.argmax(predictions, axis=1)
        
        acc = (predictions == labels).mean()
        
        results = {
            "accuracy": acc
        }
        
        torch.cuda.empty_cache()
        return results

    
    bf16_precision = torch.cuda.is_available() and model.dtype == torch.float32
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        do_eval=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": args.num_cycles},
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=bf16_precision, # Use mixed precision if a GPU is available
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        logging_steps=args.logging_steps,
        seed=args.seed,
        report_to="tensorboard",
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )


    # 5. TRAIN THE MODEL
    # ------------------------------------
    print("Starting training on the encoder...")

    trainer.train()

    # Save the final model
    trainer.save_model(f"{args.output_dir}/final_model")
    print(f"model saved to {args.output_dir}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary NER model")
    parser.add_argument("--train_file", type=str, default="/workspace/datas/few-nerd/supervised/train.binary.csv", help="Path to the training file")
    parser.add_argument("--validation_file", type=str, default="/workspace/datas/few-nerd/supervised/dev.binary.csv", help="Path to the validation file")
    parser.add_argument("--model_checkpoint", type=str, default="google/flan-t5-base", help="Model name or path")
    
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/model_dir/flan-t5-base/binary-ner-fp32-mixed", help="Path to the output directory"
    )
    
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Dropout rate for the model"
    )
    parser.add_argument(
        "--dynamic_class_weights",
        action="store_true",
        help="Whether to use dynamic class weights during training"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing value for CrossEntropyLoss"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=5,
        help="Number of cycles for cosine learning rate scheduler"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of evaluation steps with no improvement after which training will be stopped"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    '''args = parser.parse_args([
        "--model_checkpoint", "answerdotai/ModernBERT-base",
        "--train_file", "/workspace/data/dataset_construction_train/pos_tagging_train.csv",
        "--validation_file", "/workspace/data/dataset_construction_train/pos_tagging_val.csv",
        "--batch_size", "64",
        "--output_dir", "/workspace/model_dir/test",
    ])'''

    print(args)

    main(args)