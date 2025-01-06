import os
import yaml
import json
import random
import numpy as np
import torch
from pathlib import Path
from dataloader.dataset import load_data, build_vocabularies
from heuristic_baseline import VQABaselineModel, VQAExplanationGenerator
from metrics.metrics import VQAXEvaluator
# from utils.visualization import visualize_predictions

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config('./ViCLEVR-X/config/config.yaml')
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    output_dir = Path('ViCLEVR-X/outputs/baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data = load_data(config['data']['train_path'])
    val_data = load_data(config['data']['val_path'])
    test_data = load_data(config['data']['test_path'])
    
    # Build vocabularies
    print("Building vocabularies...")
    word2idx, idx2word, answer2idx, idx2answer = build_vocabularies(train_data)
    
    # Initialize baseline model
    print("Initializing baseline model...")
    baseline_model = VQABaselineModel(
        train_data=train_data,
        word2idx=word2idx,
        idx2word=idx2word,
        answer2idx=answer2idx,
        idx2answer=idx2answer
    )
    
    # Create explanation generator
    explanation_generator = VQAExplanationGenerator(baseline_model)
    
    # Initialize evaluator
    evaluator = VQAXEvaluator(device=config['model']['device'])
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = baseline_model.evaluate(val_data, config['data']['val_image_dir'], evaluator)
    print("Available metrics: ", val_results.keys())
    print(f"Validation Accuracy: {val_results['answer_accuracy']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = baseline_model.evaluate(test_data, config['data']['test_image_dir'], evaluator)
    print(f"Test Accuracy: {test_results['answer_accuracy']:.4f}")
    
    # Save results
    results = {
        'validation': val_results,
        'test': test_results
    }
    
    with open(output_dir / 'baseline_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # read results
    with open(output_dir / 'baseline_results.json', 'r') as f:
        results = json.load(f)
    print("ValidationResults:")
    print("Answer Accuracy: ", results['validation']['answer_accuracy'])
    print("BLEU-1:", results['validation']['bleu_1'])
    print("BLEU-2:", results['validation']['bleu_2'])
    print("BLEU-3:", results['validation']['bleu_3'])
    print("BLEU-4:", results['validation']['bleu_4'])
    print("BERTScore:", results['validation']['bertscore_f'])
    print("METEOR:", results['validation']['meteor'])
    print("ROUGE-L:", results['validation']['rouge_l'])
    print("CIDEr:", results['validation']['cider'])
    print("SPICE:", results['validation']['spice'])
    
    print("\nTestResults:")
    print("Answer Accuracy: ", results['test']['answer_accuracy'])
    print("BLEU-1:", results['test']['bleu_1'])
    print("BLEU-2:", results['test']['bleu_2'])
    print("BLEU-3:", results['test']['bleu_3'])
    print("BLEU-4:", results['test']['bleu_4'])
    print("BERTScore:", results['test']['bertscore_f'])
    print("METEOR:", results['test']['meteor'])
    print("ROUGE-L:", results['test']['rouge_l'])
    print("CIDEr:", results['test']['cider'])
    print("SPICE:", results['test']['spice'])

if __name__ == "__main__":
    main()