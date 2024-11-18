#!/bin/bash

# Define input and output directories
INPUT_DIR="../../data/vqax"
OUTPUT_DIR="../../data/vivqax"
mkdir -p $OUTPUT_DIR

# Run the translation pipeline
echo "Running translation pipeline..."
python pipeline.py --input_files $INPUT_DIR/vqaX_train.json $INPUT_DIR/vqaX_test.json $INPUT_DIR/vqaX_val.json --output_dir $OUTPUT_DIR