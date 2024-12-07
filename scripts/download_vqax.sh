#!/bin/bash
DATASET_DIR="../data/vqax"
mkdir -p $DATASET_DIR

echo "Downloading VQA-X dataset..."
wget -O $DATASET_DIR/vqax.zip "https://drive.google.com/uc?export=download&id=1zPexyNo_W8L-FYq6iPcERQ5cJUUJzYhl"

# Unzip the dataset
echo "Unzipping dataset..."
unzip $DATASET_DIR/vqax.zip -d $DATASET_DIR
rm $DATASET_DIR/vqax.zip

echo "Download complete."
