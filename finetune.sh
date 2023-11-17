python finetune.py \
    --pretrain_ckpt "pretrained model path" \
    --data_path "path to preprocessed data folder" \
    --reload_item_embeddings True \
    --num_train_epochs 128 \
    --batch_size 16 \
    --device 1 \
    --fp16 \
    --finetune_negative_sample_size -1