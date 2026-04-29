import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    tokenizer.model_max_length = 1024


    # Load the model for token classification
    # THIS IS THE KEY STEP: T5ForTokenClassification uses the T5 encoder ONLY.
    # The decoder is not used.
    
    label_names = ["noun", "verb"]
    # Create id2label and label2id mappings for the model
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    
    lower_cp = args.model_checkpoint.lower()
    model = None
    if "modernbert" in lower_cp:
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
    elif "t5gemma" in lower_cp:
        from transformers import T5GemmaForSequenceClassification
        model = T5GemmaForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            is_encoder_decoder=False,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    elif "t5gemma-2-" in lower_cp:
        from transformers import T5Gemma2ForSequenceClassification
        model = T5Gemma2ForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            is_encoder_decoder=False,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    
    def preprocess_function(batch_samples):
        encoded = tokenizer(batch_samples["prompt"], padding="max_length", truncation=True)
        encoded["labels"] = batch_samples["label"]
        return encoded
    
    test_dataset = HFDataset.from_csv(args.test_file)
    test_encoded_dataset = test_dataset.map(preprocess_function, batched=True, batch_size=10)
    test_encoded_dataset = test_encoded_dataset.remove_columns(["prompt", "word", "senses", "gold_image", "gold_sense", "ambiguous_sentence"])
    
    #print(test_dataset.get_label_ratio())

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(p):
        decision_threshold = args.decision_threshold
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

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.prediction_output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=False,
        do_predict=True,
        report_to="none", # Disable logging to wandb/tensorboard
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer
    )
    test_loader = DataLoader(test_encoded_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator, shuffle=False)
    print("Starting inference on the test set...")
    #predictions = trainer.predict(test_loader)
    with torch.no_grad():
        predictions = trainer.prediction_loop(test_loader, description="Prediction")
        
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))

    # 6. SAVE AND PRINT RESULTS
    print("\n--- Test Set Metrics ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    # Create the output directory if it doesn't exist
    os.makedirs(args.prediction_output_dir, exist_ok=True)
    
    # Save metrics to a JSON file
    metrics_output_path = os.path.join(args.prediction_output_dir, "test_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nTest metrics saved to: {metrics_output_path}")

    probabilities = np_softmax(predictions.predictions, axis=1)
    output_probabilities_path = os.path.join(args.prediction_output_dir, "test_probabilities.npy")
    np.save(output_probabilities_path, probabilities)
    print(f"Test probabilities saved to: {output_probabilities_path}")
    
    test_df = pd.read_csv(args.test_file)
    
    test_df["predicted_label"] = predictions.predictions.argmax(axis=1)
    
    output_csv_path = os.path.join(args.prediction_output_dir, "test_predictions.csv")
    test_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to: {output_csv_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary NER model")
    parser.add_argument("--test_file", type=str, default="/workspace/datas/few-nerd/supervised/train.binary.csv", help="Path to the training file")
    parser.add_argument("--model_checkpoint", type=str, default="google/flan-t5-base", help="Model name or path")
    
    parser.add_argument(
        "--prediction_output_dir", type=str, default="/workspace/datas/encoder/test", help="Path to the output directory"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification"
    )
    
    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "/workspace/model_dir/classification/flan-t5-base/conll2003/encoder-switch-ner-custom-drop20-cycle10-lr2e-4-cosine_restart/final_model",
        "--test_file", "/workspace/datas/mit_restaurant/test.switch.csv",
        "--custom_model",
        "--batch_size", "256",
        "--prediction_output_dir", "/workspace/datas/encoder/test"
    ])'''

    print(args)

    main(args)