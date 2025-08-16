import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import yaml
from typing import Dict, Tuple
import argparse

from src.models.baseline_model.vivqax_model import ViVQAX_Model
from src.models.baseline_model.metrics.metrics import VQAXEvaluator
from src.models.baseline_model.dataloaders.dataloader import get_dataloaders


class VQAXTrainer:
    def __init__(self, config: Dict):
        """
        Initialize the VQA-X trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['model']['device'])

        # Setup directories
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))

        # Initialize dataloaders
        self.train_loader, self.val_loader, self.test_loader, \
            self.word2idx, self.idx2word, self.answer2idx, self.idx2answer = get_dataloaders(
                config)

        # Initialize model
        self.model = ViVQAX_Model(
            vocab_size=len(self.word2idx),
            embed_size=config['model']['embed_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_answers=len(self.answer2idx),
            max_explanation_length=config['model']['max_explanation_length'],
            word2idx=self.word2idx
        ).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5
        )

        # Initialize loss functions
        self.criterion_answer = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_explanation = nn.CrossEntropyLoss(
            ignore_index=self.word2idx['<PAD>'],
            label_smoothing=0.1
        )

        # Initialize evaluator
        self.evaluator = VQAXEvaluator(device=self.device)

        self.best_val_loss = float('inf')
        self.best_metrics = {}

    def compute_loss(self,
                     answer_logits: torch.Tensor,
                     explanation_outputs: torch.Tensor,
                     answers: torch.Tensor,
                     explanations: torch.Tensor,
                     alpha: float = 0.5) -> Tuple[torch.Tensor, float, float]:
        """Compute the combined loss for answers and explanations."""
        # compute answer loss
        answer_loss = self.criterion_answer(answer_logits, answers)

        # Flatten the explanation outputs and target explanations
        # This is neccesary because the explanation outputs and target explanations are in different shapes
        # and we need to compare them in the same shape
        explanation_outputs_flat = explanation_outputs.view(
            -1, explanation_outputs.size(-1))
        explanations_flat = explanations[:, 1:].contiguous(
        ).view(-1)  # Exclude start token

        # compute explanation loss
        explanation_loss = self.criterion_explanation(
            explanation_outputs_flat, explanations_flat)

        # compute total loss use weighted sum of answer loss and explanation loss
        total_loss = alpha * answer_loss + (1 - alpha) * explanation_loss

        if not torch.isfinite(total_loss):
            raise ValueError("Loss is not finite")

        return total_loss, answer_loss.item(), explanation_loss.item()

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'answer2idx': self.answer2idx,
            'idx2answer': self.idx2answer
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')

        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            self.best_metrics = metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_answer_loss = 0
        total_explanation_loss = 0

        num_epochs = self.config['training']['num_epochs']
        initial_tf_ratio = 1.0
        final_tf_ratio = 0.1 
        teacher_forcing_ratio = max(final_tf_ratio, initial_tf_ratio - (epoch / (num_epochs * 0.8)) * (initial_tf_ratio - final_tf_ratio))
        print(f"Epoch {epoch+1} - Using Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}")
        
        train_loop = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch in train_loop:
            self.optimizer.zero_grad()

            # Move batch to device
            images = batch['image'].to(self.device)
            questions = batch['question'].to(self.device)
            answers = batch['answer'].to(self.device)
            explanations = batch['explanation'].to(self.device)

            # Forward pass
            answer_logits, explanation_outputs = self.model(
                images,
                questions,
                explanations,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            # Compute loss
            loss, answer_loss, explanation_loss = self.compute_loss(
                answer_logits, explanation_outputs, answers, explanations
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_answer_loss += answer_loss
            total_explanation_loss += explanation_loss

            train_loop.set_postfix(loss=loss.item())

        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_answer_loss = total_answer_loss / len(self.train_loader)
        avg_explanation_loss = total_explanation_loss / len(self.train_loader)

        return {
            'loss': avg_loss,
            'answer_loss': avg_answer_loss,
            'explanation_loss': avg_explanation_loss
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model efficiently in a single pass."""
        self.model.eval()
        total_loss = 0
        
        all_answer_preds = []
        all_answer_trues = []
        all_explanation_preds = {}
        all_explanation_trues = {}
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Move batch to device
                images = batch['image'].to(self.device)
                questions = batch['question'].to(self.device)
                answers = batch['answer'].to(self.device)
                explanations = batch['explanation'].to(self.device)

                # Forward pass for both loss and generation
                answer_logits, explanation_outputs_for_loss = self.model(
                    images,
                    questions,
                    explanations, # Pass target explanations for loss calculation
                    teacher_forcing_ratio=0.0
                )
                
                # 1. Compute loss
                loss, _, _ = self.compute_loss(
                    answer_logits, explanation_outputs_for_loss, answers, explanations
                )
                total_loss += loss.item()

                # Generate explanations for metric evaluation
                _, generated_explanations = self.model.generate_explanation(images, questions)

                # 2. Collect predictions and ground truths for metrics
                predicted_answers = answer_logits.argmax(dim=1)
                all_answer_preds.extend(predicted_answers.cpu().numpy())
                all_answer_trues.extend(answers.cpu().numpy())
                
                for j, (pred, true) in enumerate(zip(generated_explanations, explanations)):
                    pred_words = ' '.join([
                        self.idx2word[idx.item()] 
                        for idx in pred 
                        if self.idx2word[idx.item()] not in ['<PAD>', '<UNK>', '<START>', '<END>']
                    ])
                    true_words = ' '.join([
                        self.idx2word[idx.item()] 
                        for idx in true 
                        if self.idx2word[idx.item()] not in ['<PAD>', '<UNK>', '<START>', '<END>']
                    ])
                    sample_id = f"{i}_{j}"
                    all_explanation_preds[sample_id] = [pred_words]
                    all_explanation_trues[sample_id] = [true_words]

        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)

        answer_metrics = self.evaluator.compute_answer_metrics(
            all_answer_preds, all_answer_trues)
        
        explanation_metrics = self.evaluator.compute_explanation_metrics(
            all_explanation_preds, all_explanation_trues)

        # Combine all metrics
        metrics = {**answer_metrics, **explanation_metrics}
        metrics['loss'] = avg_loss
        
        return metrics

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']

        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            print(train_metrics)

            # Validation phase
            val_metrics = self.validate()
            print(val_metrics)

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])

            # Log metrics
            for name, value in {**train_metrics, **val_metrics}.items():
                self.writer.add_scalar(f'metrics/{name}', value, epoch)

            # Save checkpoint if best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Answer Accuracy: {val_metrics['answer_accuracy']:.4f}")
            print(f"Val BLEU-4: {val_metrics['bleu_4']:.4f}")
            print(f"Val METEOR: {val_metrics['meteor']:.4f}")
            print(f"Val CIDEr: {val_metrics['cider']:.4f}")
            print(f"Val SPICE: {val_metrics['spice']:.4f}")
            print(f"Val BERTScore: {val_metrics['bertscore_f']:.4f}")
            print("=" * 50)



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ViVQA-X model')

    # Model arguments
    parser.add_argument('--config', type=str, default='src/models/baseline_model/config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--embed_size', type=int, default=400,
                        help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--max_explanation_length', type=int, default=15,
                        help='Maximum explanation length')

    # Training arguments
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default="src/models/baseline_model/weights",
                        help='Directory to save model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Dataset paths
    parser.add_argument('--train_path', type=str,
                        default="/mnt/VLAI_data/ViVQA-X/ViVQA-X_train.json",
                        help='Path to training data file')
    parser.add_argument('--val_path', type=str,
                        default="/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json",
                        help='Path to validation data file')
    parser.add_argument('--test_path', type=str,
                        default="/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json",
                        help='Path to test data file')

    # Image directories
    parser.add_argument('--train_image_dir', type=str,
                        default='/mnt/VLAI_data/COCO_Images/train2014',
                        help='Path to training images directory')
    parser.add_argument('--val_image_dir', type=str,
                        default='/mnt/VLAI_data/COCO_Images/val2014',
                        help='Path to validation images directory')
    parser.add_argument('--test_image_dir', type=str,
                        default='/mnt/VLAI_data/COCO_Images/val2014',
                        help='Path to test images directory')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override model config
    if args.device:
        config['model']['device'] = args.device
    if args.embed_size:
        config['model']['embed_size'] = args.embed_size
    if args.hidden_size:
        config['model']['hidden_size'] = args.hidden_size
    if args.num_layers:
        config['model']['num_layers'] = args.num_layers
    if args.max_explanation_length:
        config['model']['max_explanation_length'] = args.max_explanation_length

    # Override training config
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_workers:
        config['training']['num_workers'] = args.num_workers
    if args.save_dir:
        config['training']['save_dir'] = args.save_dir
    if args.seed:
        config['training']['seed'] = args.seed

    # Override data paths
    if args.train_path:
        config['data']['train_path'] = args.train_path
    if args.val_path:
        config['data']['val_path'] = args.val_path
    if args.test_path:
        config['data']['test_path'] = args.test_path

    # Override image directories
    if args.train_image_dir:
        config['data']['train_image_dir'] = args.train_image_dir
    if args.val_image_dir:
        config['data']['val_image_dir'] = args.val_image_dir
    if args.test_image_dir:
        config['data']['test_image_dir'] = args.test_image_dir

    # Set random seed
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])

    trainer = VQAXTrainer(config)

    # Start training
    try:
        trainer.train()
    finally:
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
