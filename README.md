# ViVQA-X Models

This repository contains two different models for the ViVQA-X task:
1. Baseline Model - A deep learning based approach
2. Heuristic Model - A rule-based approach

## Project Structure

```
src/models/
├── baseline_model/
│   ├── vivqax_model.py    # Main model implementation
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── config -> ../config    # Symlink to shared config
├── heuristic_model/
│   ├── heuristic_baseline.py  # Heuristic model implementation
│   ├── run_heuristic.py       # Script to run heuristic model
│   └── config -> ../config    # Symlink to shared config
├── config/                # Shared configuration
├── dataloader/           # Shared data loading utilities
├── metrics/              # Shared evaluation metrics
└── utils/                # Shared utility functions
```

## Prerequisites

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data according to the configuration in `src/models/config/config.yaml`

## Running the Baseline Model

The baseline model is a deep learning based approach that requires training.

### Training

To train the baseline model:

```bash
cd src/models/baseline_model
python train.py [arguments]
```

Available arguments:
- `--config`: Path to config file (default: ./config/config.yaml)
- `--device`: Device to use (cuda/cpu) (default: cuda:2)
- `--embed_size`: Embedding size (default: 400)
- `--hidden_size`: Hidden size (default: 2048)
- `--num_layers`: Number of layers (default: 2)
- `--max_explanation_length`: Maximum explanation length (default: 15)
- `--lr`: Learning rate (default: 0.0001)
- `--num_epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size (default: 128)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--save_dir`: Directory to save model weights
- `--seed`: Random seed (default: 0)

### Evaluation

To evaluate the trained baseline model:

```bash
cd src/models/baseline_model
python evaluate.py --model_path path/to/saved/model
```

## Running the Heuristic Model

The heuristic model is a rule-based approach that doesn't require training.

To run the heuristic model:

```bash
cd src/models/heuristic_model
python run_heuristic.py
```

The script will:
1. Load the data
2. Initialize the heuristic model
3. Evaluate on validation and test sets
4. Save results to `outputs/baseline/baseline_results.json`

## Configuration

Both models share configuration from `src/models/config/config.yaml`. Key configurations include:

```yaml
model:
  device: "cuda:2"
  embed_size: 400
  hidden_size: 2048
  num_layers: 2
  max_explanation_length: 15

training:
  learning_rate: 0.0001
  num_epochs: 10
  batch_size: 128
  num_workers: 4

data:
  train_path: "path/to/train.json"
  val_path: "path/to/val.json"
  test_path: "path/to/test.json"
  train_image_dir: "path/to/train/images"
  val_image_dir: "path/to/val/images"
  test_image_dir: "path/to/test/images"
```

## Results

The evaluation results will include:
- Answer Accuracy
- BLEU scores (1-4)
- BERTScore
- METEOR
- ROUGE-L
- CIDEr
- SPICE

Results are saved in JSON format for both validation and test sets.
