# An Automated Pipeline for Constructing a Vietnamese VQA-NLE Dataset

This repository contains the code and resources for the paper "An Automated Pipeline for Constructing a Vietnamese VQA-NLE Dataset".

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Pipeline](#pipeline)
  - [Benchmark](#benchmark)
    - [Heuristic Model](#heuristic-model)
    - [Baseline Model](#baseline-model)
- [Directory Structure](#directory-structure)

## Introduction

This project provides an automated pipeline for constructing a Vietnamese Visual Question Answering with Natural Language Explanations (VQA-NLE) dataset. The dataset includes images, questions, answers, and explanations in Vietnamese, facilitating research in the field of VQA and NLE.

## Dataset

The dataset is organized into several JSON files located in the `data/final` directory. Each file contains a collection of questions, answers, and explanations associated with images from the COCO dataset.

You can also access the dataset on Hugging Face: [ViVQA-X Dataset on Hugging Face](https://huggingface.co/datasets/duongtruongbinh/ViVQA-X)

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/duongtruongbinh/ViVQA-X.git
   cd ViVQA-X
   ```

2. Create a virtual environment and activate it:

   ```sh
   conda create -n vivqa-x_env
   conda activate vivqa-x_env
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

4. Download the dataset:
   ```sh
   bash scripts/download_vqax.sh
   ```

## Usage

### Pipeline

To run the pipeline, execute the following command:

```sh
bash scripts/pipeline.sh
```

### Benchmark

We assess the performance of several models on the ViVQA-X dataset, including:

- Heuristic Model
- LSTM-Generative (Baseline Model)
- [NLX-GPT](https://github.com/fawazsammani/nlxgpt)
- [OFA-X](https://github.com/ofa-x/OFA-X)
- [ReRe](https://github.com/yeonsue/ReRe)

For the models NLX-GPT, OFA-X, and ReRe, please refer to their respective repositories for detailed evaluation instructions on the ViVQA-X dataset.

#### Heuristic Model

The heuristic model is a rule-based approach that doesn't require training. To run the heuristic model:

1. Configure the model settings in `src/models/heuristic_model/config/config.yaml`:
   ```yaml
   data:
     train_path: "data/final/ViVQA-X_train.json"
     val_path: "data/final/ViVQA-X_val.json"
     test_path: "data/final/ViVQA-X_test.json"
     train_image_dir: "path/to/train/images"
     val_image_dir: "path/to/val/images"
     test_image_dir: "path/to/test/images"
   ```

2. Run the heuristic model:
   ```sh
   python src/models/heuristic_model/run_heuristic.py
   ```

The script will evaluate the model on both validation and test sets and save results to `outputs/baseline/baseline_results.json`.

#### Baseline Model

The baseline model (LSTM-Generative) requires training before evaluation. Follow these steps:

1. Configure the model settings in `src/models/baseline_model/config/config.yaml`. Key configurations include:
   ```yaml
   model:
     device: "cuda:0"  # Adjust based on your GPU availability
     embed_size: 400
     hidden_size: 2048
     num_layers: 2
     max_explanation_length: 15

   training:
     learning_rate: 0.0001
     num_epochs: 10
     batch_size: 128
     num_workers: 4
     save_dir: "weights/baseline"
   ```

2. Train the model:
   ```sh
   python src/models/baseline_model/train.py [arguments]
   ```
   Available arguments:
   - `--config`: Path to config file (default: ./config/config.yaml)
   - `--device`: Device to use (cuda/cpu)
   - `--embed_size`: Embedding size
   - `--hidden_size`: Hidden size
   - `--num_layers`: Number of layers
   - `--lr`: Learning rate
   - `--num_epochs`: Number of epochs
   - `--batch_size`: Batch size
   - `--save_dir`: Directory to save model weights

3. Evaluate the trained model:
   ```sh
   python src/models/baseline_model/evaluate.py --model_path path/to/saved/model
   ```

Both models will output evaluation metrics including:
- Answer Accuracy
- BLEU scores (1-4)
- BERTScore
- METEOR
- ROUGE-L
- CIDEr
- SPICE

## Directory Structure

The directory structure of the project is as follows:

```
.
├── data/
│   ├── vqax
│   ├── translation
│   ├── selection
│   └── final
├── notebooks/
├── scripts/
│   ├── download_vqax.sh
│   ├── pipeline.sh
│   ├── train.sh
│   └── evaluate.sh
├── src/
│   ├── models/
│   │   ├── baseline_model/
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── vivqax_model.py
│   │   └── heuristic_model/
│   │       ├── run_heuristic.py
│   │       └── heuristic_baseline.py
│   ├── pipeline/
│   │   ├── translation/
│   │   ├── selection/
│   │   ├── post_processing/
│   │   └── pipeline.py
├── requirements.txt
└── README.md
```

## Citation

```
