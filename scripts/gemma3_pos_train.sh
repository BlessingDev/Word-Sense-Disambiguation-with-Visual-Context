
python /workspace/generation_decoder_train.py \
    --model_checkpoint google/gemma-3-4b-it \
    --train_file /workspace/data/dataset_construction_train/train_set_pos.csv \
    --validation_file /workspace/data/dataset_construction_train/val_set_pos.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1 \
    --output_dir /workspace/model_dir/gemma3-4b-it/pos-descriminator/ \
    --train_epochs 10 \
    --weight_decay 0.01 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_steps 30 \
    --logging_steps 10