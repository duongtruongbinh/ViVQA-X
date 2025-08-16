# An Automated Pipeline for Constructing a Vietnamese VQA-NLE Dataset

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/VLAI-AIVN/ViVQA-X_LSTM-Generative)
[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-orange)](https://huggingface.co/spaces/VLAI-AIVN/ViVQA-X_LSTM-Generative_Demo)


This repository contains the code and resources for the paper "An Automated Pipeline for 
Constructing a Vietnamese VQA-NLE Dataset".

## Table of Contents

- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Quick Start](#-quick-start)
- [Installation](#️-installation)
- [Usage](#-usage)
  - [Pipeline](#-pipeline)
  - [Benchmark](#-benchmark)
    - [Heuristic Model](#-heuristic-model)
    - [Baseline Model](#-baseline-model)
- [Directory Structure](#-directory-structure)
- [Citation](#-citation)
- [License](#-license)

## Introduction

This project introduces **ViVQA-X**, the first Vietnamese dataset for Visual Question Answering with Natural Language Explanations (VQA-NLE). Developed using a novel automated pipeline, our work provides a crucial resource to advance research in multimodal AI and explainability for the Vietnamese language. **ViVQA-X** features:

- **32,886** question-answer pairs with detailed explanations
- **41,817** high-quality natural language explanations
- Multi-stage automated pipeline for translation and quality control
- Comprehensive evaluation using multiple state-of-the-art models

This project facilitates research in Vietnamese visual question answering and supports the development of explainable AI systems for Vietnamese language understanding.

## Dataset

### 🔗 Access Points

| Resource | Description | Link |
|----------|-------------|------|
| **Dataset** | ViVQA-X Dataset on Hugging Face | [![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X) |
| **Model Weights** | Pre-trained LSTM-Generative Model | [![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/VLAI-AIVN/ViVQA-X_LSTM-Generative) |
| **Demo** | Interactive Demo Space | [![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-orange)](https://huggingface.co/spaces/VLAI-AIVN/ViVQA-X_LSTM-Generative_Demo) |

### 📈 Dataset Statistics

- **QA Pairs**: 32,886 pairs across Train/Validation/Test splits
- **Explanations**: 41,817 high-quality explanations  
- **Average Words**: 10 words per explanation
- **Vocabulary Size**: 4,232 unique words in explanations
- **Images**: COCO dataset images with Vietnamese annotations

The dataset is organized into JSON files located in the `data/final` directory, containing questions, answers, and explanations associated with images from the COCO dataset.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/duongtruongbinh/ViVQA-X.git
cd ViVQA-X

# Install dependencies
pip install -r requirements.txt

# Download the dataset
bash scripts/download_vqax.sh

# Run the complete pipeline
bash scripts/pipeline.sh
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.2+ (for GPU support)
- 8GB+ RAM recommended

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/duongtruongbinh/ViVQA-X.git
   cd ViVQA-X
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using conda (recommended)
   conda create -n vivqa-x python=3.8
   conda activate vivqa-x
   
   # Or using venv
   python -m venv vivqa-x
   source vivqa-x/bin/activate  # On Windows: vivqa-x\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (for pipeline)**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env file and add your API keys:
   # OPENAI_API_KEY=your_openai_api_key_here
   # GEMINI_APIKEYS=your_gemini_api_key_1,your_gemini_api_key_2
   ```

5. **Download the original VQA-X dataset**
   ```bash
   bash scripts/download_vqax.sh
   ```

6. **Download the COCO dataset**
   The ViVQA-X dataset uses images from the COCO 2014 dataset. You need to download the `train2014` and `val2014` image sets.
   ```bash
   # Create directory for COCO data
   mkdir -p data/coco

   # Download and unzip Train 2014 images (~13GB)
   wget http://images.cocodataset.org/zips/train2014.zip -P data/coco/
   unzip data/coco/train2014.zip -d data/coco/
   rm data/coco/train2014.zip

   # Download and unzip Validation 2014 images (~6GB)
   wget http://images.cocodataset.org/zips/val2014.zip -P data/coco/
   unzip data/coco/val2014.zip -d data/coco/
   rm data/coco/val2014.zip
   ```

   After this step, you should have the following directory structure: `data/coco/train2014` and `data/coco/val2014`.

## Usage

### Pipeline

Run the complete translation and processing pipeline:

```bash
bash scripts/pipeline.sh
```

This will:
- Translate English VQA-X to Vietnamese
- Apply quality selection mechanisms
- Post-process the results
- Generate the final ViVQA-X dataset

### Benchmark

We provide comprehensive benchmarks using multiple state-of-the-art models:

| Model | Repository |
|-------|------|
| **Heuristic Model** | Included |
| **LSTM-Generative** | Included |
| **NLX-GPT** | [GitHub](https://github.com/fawazsammani/nlxgpt) |
| **OFA-X** | [GitHub](https://github.com/ofa-x/OFA-X) |
| **ReRe** | [GitHub](https://github.com/yeonsue/ReRe) |

#### Heuristic Model

A rule-based approach requiring no training:

1. **Configure the model**
   ```yaml
   # src/models/heuristic_model/config/config.yaml
   data:
     train_path: "data/final/ViVQA-X_train.json"
     val_path: "data/final/ViVQA-X_val.json"
     test_path: "data/final/ViVQA-X_test.json"
     train_image_dir: "data/coco/train2014"
     val_image_dir: "data/coco/val2014"
     test_image_dir: "data/coco/val2014"
   ```

2. **Run evaluation**
   ```bash
   python src/models/heuristic_model/run_heuristic.py
   ```

#### Baseline Model

LSTM-Generative model with attention mechanism:

1. **Configure the model**
   ```yaml
   # src/models/baseline_model/config/config.yaml
   data:
     train_path: "data/final/ViVQA-X_train.json"
     val_path: "data/final/ViVQA-X_val.json"
     test_path: "data/final/ViVQA-X_test.json"
     train_image_dir: 'data/coco/train2014'
     val_image_dir: 'data/coco/val2014'
     test_image_dir: 'data/coco/val2014'

   model:
     device: "cuda:0"  # Adjust based on GPU availability
     embed_size: 400
     hidden_size: 2048
     num_layers: 2
     max_explanation_length: 15

   training:
     learning_rate: 0.0001
     num_epochs: 50
     batch_size: 128
     num_workers: 4
     save_dir: "weights/baseline"
   ```

2. **Train the model**
   ```bash
   # Using script (recommended)
   bash scripts/train.sh
   
   # Or direct command
   python src/models/baseline_model/train.py --config src/models/baseline_model/config/config.yaml
   ```

3. **Evaluate the model**
   ```bash
   # Using script
   bash scripts/evaluate.sh
   
   # Or direct command  
   python src/models/baseline_model/evaluate.py --model_path weights/baseline/best_model.pth
   ```

4. **Use pre-trained weights**
   
   Download from [![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/VLAI-AIVN/ViVQA-X_LSTM-Generative):
   ```bash
   # The model weights are available on Hugging Face
   # Follow the repository instructions to download and use
   ```

### Evaluation Metrics

Both models provide comprehensive evaluation metrics:

| Metric | Description |
|--------|-------------|
| **Answer Accuracy** | Exact match accuracy for answers |
| **BLEU-1/2/3/4** | N-gram precision for explanations |
| **BERTScore** | Contextual similarity score |
| **METEOR** | Semantic similarity with WordNet |
| **ROUGE-L** | Longest common subsequence |
| **CIDEr** | Consensus-based evaluation |
| **SPICE** | Semantic propositional evaluation |

## Directory Structure

```
ViVQA-X/
├── data/                          # Dataset files
│   ├── vqax/                         # Original VQA-X dataset
│   ├── translation/                  # Translation intermediate files
│   ├── selection/                    # Quality selection files
│   └── final/                        # Final ViVQA-X dataset
│       ├── ViVQA-X_train.json
│       ├── ViVQA-X_val.json
│       └── ViVQA-X_test.json
├── notebooks/                     # Jupyter notebooks for analysis
├── scripts/                       # Utility scripts
│   ├── download_vqax.sh              # Download original dataset
│   ├── pipeline.sh                   # Run complete pipeline
│   ├── train.sh                      # Train baseline model
│   └── evaluate.sh                   # Evaluate models
├── src/                           # Source code
│   ├── models/                       # Model implementations
│   │   ├── baseline_model/           # LSTM-Generative model
│   │   │   ├── config/               # Configuration files
│   │   │   ├── dataloaders/          # Data loading utilities
│   │   │   ├── metrics/              # Evaluation metrics
│   │   │   ├── utils/                # Helper utilities
│   │   │   ├── weights/              # Model checkpoints
│   │   │   ├── train.py              # Training script
│   │   │   ├── evaluate.py           # Evaluation script
│   │   │   └── vivqax_model.py       # Model architecture
│   │   └── heuristic_model/          # Rule-based baseline
│   │       ├── config/               # Configuration files
│   │       ├── dataloaders/          # Data loading utilities
│   │       ├── metrics/              # Evaluation metrics
│   │       ├── utils/                # Helper utilities
│   │       ├── run_heuristic.py      # Main evaluation script
│   │       └── heuristic_baseline.py # Model implementation
│   └── pipeline/                     # Data processing pipeline
│       ├── translation/              # Translation modules
│       │   ├── translators/          # Various translator implementations
│       │   └── translation.py        # Translation pipeline
│       ├── selection/                # Quality selection modules
│       │   ├── evaluators/           # LLM evaluators
│       │   └── selection.py          # Selection pipeline
│       ├── post_processing/          # Post-processing modules
│       └── pipeline.py               # Main pipeline script
├── requirements.txt               # Python dependencies
├── LICENSE                        
└── README.md                      # This file
```

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@misc{vivqax2025,
  author    = {Duong, Truong-Binh and Tran, Hoang-Minh and Le-Nguyen, Binh-Nam and Duong, Dinh-Thang},
  title     = {An Automated Pipeline for Constructing a Vietnamese VQA-NLE Dataset},
  howpublished = {Accepted for publication in the Proceedings of The International Conference on Intelligent Systems \& Networks (ICISN 2025), Springer Lecture Notes in Networks and Systems (LNNS)},
  year      = {2025}
}
```


<div align="center">
  <p>
    <a href="https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X">🤗 Dataset</a> •
    <a href="https://huggingface.co/VLAI-AIVN/ViVQA-X_LSTM-Generative">🤗 Model</a> •
    <a href="https://huggingface.co/spaces/VLAI-AIVN/ViVQA-X_LSTM-Generative_Demo">🤗 Demo</a> •
    <a href="mailto:duongtruongbinh@gmail.com">📧 Contact</a>
  </p>
</div>
