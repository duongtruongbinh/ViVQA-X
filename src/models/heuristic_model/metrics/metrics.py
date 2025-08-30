from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import time
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import BERTScorer
from sklearn.metrics import accuracy_score, f1_score
import torch

class VQAXEvaluator:
    """
    Evaluator class for VQA-X that computes multiple metrics:
    - Answer Accuracy and F1
    - BLEU (1-4)
    - ROUGE-L
    - CIDEr
    - METEOR
    - SPICE
    - BERTScore
    """
    def __init__(self, device: str = "cuda:0"):
        """
        Initialize the evaluator with all necessary metric calculators.
        
        Args:
            device (str): Device to use for BERTScore computation
        """
        self.device = device
        
        # Initialize all scorers
        self.scorers = {
            'Bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'METEOR': (Meteor(), "METEOR"),
            'CIDEr': (Cider(), "CIDEr"),
            'ROUGE_L': (Rouge(), "ROUGE_L"),
            'SPICE': (Spice(), "SPICE")
        }
        
        # Initialize BERTScore
        self.bert_scorer = BERTScorer(
            lang="vi",
            rescale_with_baseline=False,
            device=device
        )

    def _prepare_explanation_data(self, 
                                predictions: Dict[str, List[str]], 
                                references: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
        """
        Prepare explanation data for evaluation.
        
        Args:
            predictions: Dictionary of predicted explanations
            references: Dictionary of reference explanations
            
        Returns:
            Tuple of processed predictions and references
        """
        # Filter out empty predictions/references
        valid_ids = [k for k in predictions.keys() 
                    if len(predictions[k][0].strip()) > 0 and 
                    len(references[k][0].strip()) > 0]
        
        processed_preds = {k: predictions[k] for k in valid_ids}
        processed_refs = {k: references[k] for k in valid_ids}
        
        return processed_preds, processed_refs

    def compute_answer_metrics(self, 
                             predicted_answers: List[int], 
                             ground_truth_answers: List[int]) -> Dict[str, float]:
        """
        Compute metrics for answer prediction.
        
        Args:
            predicted_answers: List of predicted answer indices
            ground_truth_answers: List of ground truth answer indices
            
        Returns:
            Dictionary containing accuracy and F1 score
        """
        accuracy = accuracy_score(ground_truth_answers, predicted_answers)
        f1 = f1_score(ground_truth_answers, predicted_answers, average='weighted')
        
        return {
            'answer_accuracy': accuracy,
            'answer_f1_score': f1
        }

    def compute_explanation_metrics(self, 
                                  predictions: Dict[str, List[str]], 
                                  references: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute all metrics for explanation generation.
        
        Args:
            predictions: Dictionary of predicted explanations
            references: Dictionary of reference explanations
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Prepare data
        processed_preds, processed_refs = self._prepare_explanation_data(
            predictions, references)
        
        if not processed_preds:
            print("Warning: No valid prediction-reference pairs found!")
            return {metric: 0.0 for metric in 
                   ["bleu1", "bleu2", "bleu3", "bleu4", 
                    "meteor", "cider", "rouge_l", "spice", 
                    "bertscore_p", "bertscore_r", "bertscore_f"]}
        
        # Compute traditional metrics
        for scorer_name, (scorer, method) in self.scorers.items():
            if scorer_name == 'SPICE':
                t_start = time.time()
            
            score, scores = scorer.compute_score(processed_refs, processed_preds)
            
            if scorer_name == 'SPICE':
                print(f"SPICE evaluation took: {time.time() - t_start:.2f} s")
            
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    metrics[m.lower()] = sc
            else:
                metrics[method.lower()] = score
        
        # Compute BERTScore
        all_preds = [pred[0] for pred in processed_preds.values()]
        all_refs = [ref[0] for ref in processed_refs.values()]
        
        P, R, F1 = self.bert_scorer.score(all_preds, all_refs)
        
        metrics.update({
            'bertscore_p': P.mean().item(),
            'bertscore_r': R.mean().item(),
            'bertscore_f': F1.mean().item()
        })
        
        return metrics

    def evaluate(self, 
                model: torch.nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                idx2word: Dict[int, str]) -> Dict[str, float]:
        """
        Evaluate a model using all metrics.
        
        Args:
            model: The VQA-X model to evaluate
            data_loader: DataLoader containing evaluation data
            idx2word: Dictionary mapping indices to words
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        model.eval()
        all_answer_preds = []
        all_answer_trues = []
        all_explanation_preds = {}
        all_explanation_trues = {}
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                # Move batch to device
                images = batch['image'].to(self.device)
                questions = batch['question'].to(self.device)
                answers = batch['answer'].to(self.device)
                explanations = batch['explanation'].to(self.device)
                
                # Get model predictions
                answer_logits, generated_explanations = model.generate_explanation(
                    images, questions)
                
                # Process answers
                predicted_answers = answer_logits.argmax(dim=1)
                all_answer_preds.extend(predicted_answers.cpu().numpy())
                all_answer_trues.extend(answers.cpu().numpy())
                
                # Process explanations
                for j, (pred, true) in enumerate(zip(generated_explanations, explanations)):
                    # Convert indices to words and join them
                    pred_words = ' '.join([
                        idx2word[idx.item()] 
                        for idx in pred 
                        if idx2word[idx.item()] not in ['<PAD>', '<UNK>', '<START>', '<END>']
                    ])
                    
                    true_words = ' '.join([
                        idx2word[idx.item()] 
                        for idx in true 
                        if idx2word[idx.item()] not in ['<PAD>', '<UNK>', '<START>', '<END>']
                    ])
                    
                    sample_id = f"{i}_{j}"
                    all_explanation_preds[sample_id] = [pred_words]
                    all_explanation_trues[sample_id] = [true_words]
        
        # Compute all metrics
        answer_metrics = self.compute_answer_metrics(
            all_answer_preds, all_answer_trues)
        explanation_metrics = self.compute_explanation_metrics(
            all_explanation_preds, all_explanation_trues)
        
        # Combine all metrics
        metrics = {**answer_metrics, **explanation_metrics}
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Answer Accuracy: {metrics['answer_accuracy']:.4f}")
        print(f"Answer F1-Score: {metrics['answer_f1_score']:.4f}")
        print(f"Explanation BLEU-1: {metrics['bleu_1']:.4f}")
        print(f"Explanation BLEU-2: {metrics['bleu_2']:.4f}")
        print(f"Explanation BLEU-3: {metrics['bleu_3']:.4f}")
        print(f"Explanation BLEU-4: {metrics['bleu_4']:.4f}")
        print(f"Explanation METEOR: {metrics['meteor']:.4f}")
        print(f"Explanation CIDEr: {metrics['cider']:.4f}")
        print(f"Explanation ROUGE-L: {metrics['rouge_l']:.4f}")
        print(f"Explanation SPICE: {metrics['spice']:.4f}")
        print(f"Explanation BERTScore F1: {metrics['bertscore_f']:.4f}")
        
        return metrics