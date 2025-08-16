#!/bin/bash

python src/models/baseline_model/evaluate.py \
    --checkpoint weights/baseline/best_model.pth \
    --config src/models/baseline_model/config/config.yaml
