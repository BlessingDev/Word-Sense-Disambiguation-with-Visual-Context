
python /workspace/generation_decoder_train.py \
    --model_checkpoint Qwen/Qwen3-VL-4B-Instruct \
    --attn_implementation eager \
    --train_file /workspace/data/train_set_process/wsd_set_entire_ambiguous_sentence_train.csv \
    --validation_file /workspace/data/train_set_process/wsd_set_entire_ambiguous_sentence_val.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1 \
    --output_dir /workspace/model_dir/Qwen3-VL-4B-Instruct/iwsd/ \
    --train_epochs 10 \
    --weight_decay 0.01 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_steps 30 \
    --logging_steps 50