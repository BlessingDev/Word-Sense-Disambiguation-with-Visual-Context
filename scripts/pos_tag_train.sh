# answerdotai/ModernBERT-base
# google/t5gemma-l-l-ul2-it

python /workspace/encoder_seq_classification_train.py \
    --model_checkpoint google/t5gemma-l-l-ul2-it \
    --train_file /workspace/data/dataset_construction_train/pos_tagging_train_augmentedx2.csv \
    --validation_file /workspace/data/dataset_construction_train/pos_tagging_val.csv \
    --output_dir /workspace/model_dir/t5gemma-l-l-ul2-it/pos-tagging-classifier-augmentedx2 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --weight_decay 0.0