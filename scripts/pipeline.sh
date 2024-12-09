#!/bin/bash

# Define input and output directories
INPUT_DIR="../../data/vqax"
OUTPUT_DIR="../../data/vivqax"
mkdir -p $OUTPUT_DIR

# Run the pipeline
echo "Running the pipeline..."
python src/pipeline/pipeline.py \
    --input_files $INPUT_DIR/vqaX_train.json $INPUT_DIR/vqaX_test.json $INPUT_DIR/vqaX_val.json \
    --output_dir $OUTPUT_DIR ${@:1}
# --only_translation --only_selection --only_post_processing # if you want to run only one of the phases, remove the other two flags
