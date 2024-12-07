# An Automated Pipeline for Constructing a Vietnamese VQA-NLE Dataset

This repository contains the code and resources for the paper "An Automated Pipeline for Constructing a Vietnamese VQA-NLE Dataset".

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Pipeline](#pipeline)
  - [Benchmark](#benchmark)
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
- LSTM-Generative
- [NLX-GPT](https://github.com/fawazsammani/nlxgpt)
- [OFA-X](https://github.com/ofa-x/OFA-X)
- [ReRe](https://github.com/yeonsue/ReRe)

For the models NLX-GPT, OFA-X, and ReRe, please refer to their respective repositories for detailed evaluation instructions on the ViVQA-X dataset. For the Heuristic Model and LSTM-Generative, follow these steps:

1. Run the training script:

   ```sh
   bash scripts/train.sh
   ```

2. Run the evaluation script:

   ```sh
   bash scripts/evaluate.sh
   ```

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

```
