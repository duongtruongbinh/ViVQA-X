import torch
import yaml
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List

from baseline.vivqax_model import ViVQAX_Model
from metrics.metrics import VQAXEvaluator
from dataloader.dataloader import get_dataloaders
# from utils.visualization import visualize_predictions

class VQAX_Evaluator:
    def __init__(self, checkpoint_path: str, config_path: str):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to the saved model checkpoint
            config_path: Path to the configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Setup device
        self.device = torch.device(self.config['model']['device'])
        
        # Initialize dataloaders
        self.train_loader, self.val_loader, self.test_loader, \
        self.word2idx, self.idx2word, self.answer2idx, self.idx2answer = get_dataloaders(self.config)
        
        # Initialize model
        self.model = ViVQAX_Model(
            vocab_size=len(self.word2idx),
            embed_size=self.config['model']['embed_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_answers=len(self.answer2idx),
            max_explanation_length=self.config['model']['max_explanation_length'],
            word2idx=self.word2idx
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize metrics calculator
        self.metrics_calculator = VQAXEvaluator()
        
        # Setup output directory
        self.output_dir = Path('ViCLEVR-X/evaluation_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_split(self, split: str = 'test') -> Dict:
        """
        Evaluate model on a specific data split.
        
        Args:
            split: Data split to evaluate on ('train', 'val', or 'test')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        loader = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }[split]
        
        print(f"\nEvaluating on {split} set...")
        metrics = self.metrics_calculator.evaluate(
            self.model,
            loader,
            self.idx2word
        )
        
        # Save metrics
        metrics_path = self.output_dir / f'{split}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

    def generate_visualizations(self, split: str = 'test', num_samples: int = 10):
        """Generate and save visualization samples."""
        loader = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }[split]
        
        print(f"\nGenerating visualizations for {split} set...")
        visualize_predictions(
            model=self.model,
            data_loader=loader,
            idx2word=self.idx2word,
            idx2answer=self.idx2answer,
            num_samples=num_samples,
            output_dir=self.output_dir / f'{split}_visualizations'
        )

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        # Evaluate on all splits
        splits = ['train', 'val', 'test']
        all_metrics = {}
        
        for split in splits:
            metrics = self.evaluate_split(split)
            all_metrics[split] = metrics
            
            print(f"\n{split.capitalize()} Set Metrics:")
            print(f"Answer Accuracy: {metrics['answer_accuracy']:.4f}")
            print(f"BLEU-4: {metrics['bleu_4']:.4f}")
            print(f"METEOR: {metrics['meteor']:.4f}")
            print(f"CIDEr: {metrics['cider']:.4f}")
            print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
            print(f"SPICE: {metrics['spice']:.4f}")
            print(f"BERTScore F1: {metrics['bertscore_f']:.4f}")
        
        # Generate visualizations
        # self.generate_visualizations('test', num_samples=20)
        
        # Save complete results
        with open(self.output_dir / 'all_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=4)

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to config file')
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = VQAX_Evaluator(args.checkpoint, args.config)
    
    # Run evaluation
    evaluator.run_full_evaluation()

if __name__ == '__main__':
    main()