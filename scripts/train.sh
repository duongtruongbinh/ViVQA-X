#!/bin/bash

python src/models/baseline_model/train.py \
    --config src/models/baseline_model/config/config.yaml \
    --device cuda:0 \
    --save_dir weights/baseline \
    --seed 0 \
    --embed_size 400 \
    --hidden_size 2048 \
    --num_layers 2 \
    --max_explanation_length 15 \
    --lr 0.0001 \
    --num_epochs 50 \
    --batch_size 128 \
    --num_workers 4
