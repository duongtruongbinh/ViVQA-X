import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path
import yaml
from typing import Dict, Tuple

from models.vivqax_model import ViVQAX_Model
from metrics.metrics import VQAXEvaluator
from dataloader.dataloader import get_dataloaders


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
                teacher_forcing_ratio=0.5
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
        """Validate the model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
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
                    teacher_forcing_ratio=0.0  # No teacher forcing during validation
                )

                # Compute loss
                loss, _, _ = self.compute_loss(
                    answer_logits, explanation_outputs, answers, explanations
                )
                total_loss += loss.item()

        # Get average loss
        avg_loss = total_loss / len(self.val_loader)

        # Get other metrics
        metrics = self.evaluator.evaluate(
            self.model,
            self.val_loader,
            self.idx2word
        )

        # Add loss to metrics
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


def release_memory(trainer):
    del model
    del train_loader
    del val_loader
    del test_loader
    torch.cuda.empty_cache()
    gc.collect()


def main():
    # Load config
    with open('./ViCLEVR-X/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])

    # Initialize trainer
    trainer = VQAXTrainer(config)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
