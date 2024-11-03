import random
import torch
from PIL import Image
from torchvision import transforms
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from metrics.metrics import VQAXEvaluator
from typing import Dict, List, Tuple, Any


class VQABaselineModel:
    """
    A simple baseline model for VQA that randomly samples answer-explanation pairs
    from the training set.
    """

    def __init__(self, train_data: Dict, word2idx: Dict, idx2word: Dict,
                 answer2idx: Dict, idx2answer: Dict):
        """
        Initialize the baseline model.

        Args:
            train_data (Dict): Training data dictionary
            word2idx (Dict): Word to index mapping
            idx2word (Dict): Index to word mapping
            answer2idx (Dict): Answer to index mapping
            idx2answer (Dict): Index to answer mapping
        """
        self.train_data = train_data
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.answer2idx = answer2idx
        self.idx2answer = idx2answer

        # Cache all answer-explanation pairs from training data
        self.answer_explanation_pairs = self._cache_answer_explanation_pairs()

        # Initialize transform for images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _cache_answer_explanation_pairs(self) -> List[Tuple[str, str]]:
        """Cache all valid answer-explanation pairs from training data."""
        pairs = []
        for item in self.train_data.values():
            if 'answers' in item and 'explanation' in item:
                answers = [ans['answer'] for ans in item['answers']]
                if answers and item['explanation']:
                    # Use the first answer and explanation
                    pairs.append((answers[0], item['explanation'][0]))
        return pairs

    def predict(self, image_path: str, question: str) -> Tuple[str, str]:
        """
        Make a prediction by randomly sampling an answer-explanation pair.

        Args:
            image_path (str): Path to the image
            question (str): Question text

        Returns:
            Tuple[str, str]: Predicted answer and explanation
        """
        # Randomly select an answer-explanation pair
        answer, explanation = random.choice(self.answer_explanation_pairs)
        return answer, explanation

    def evaluate(self, test_data: Dict, test_img_dir: str, evaluator: VQAXEvaluator) -> Dict[str, float]:
        """
        Evaluate the baseline model using comprehensive metrics.

        Args:
            test_data (Dict): Test dataset
            test_img_dir (str): Directory containing test images
            evaluator (VQAXEvaluator): Evaluator instance for computing metrics

        Returns:
            Dict[str, float]: Dictionary containing all evaluation metrics
        """
        predictions = {}
        references = {}
        all_pred_answers = []
        all_true_answers = []

        print("Evaluating baseline model...")
        for idx, (question_id, item) in enumerate(tqdm(test_data.items())):
            # Get ground truth
            ground_truth_answers = [ans['answer'].lower()
                                    for ans in item['answers']]
            ground_truth_explanation = item['explanation'][0] if item['explanation'] else ""

            # Make prediction
            image_path = f"{test_img_dir}/{item['image_name']}"
            pred_answer, pred_explanation = self.predict(
                image_path, item['question'])

            # Store predictions and references
            sample_id = f"sample_{idx}"
            predictions[sample_id] = [pred_explanation]
            references[sample_id] = [ground_truth_explanation]

            # Store answers
            all_pred_answers.append(
                self.answer2idx.get(pred_answer.lower(), 0))
            all_true_answers.append(self.answer2idx.get(
                ground_truth_answers[0].lower(), 0))

        # Compute all metrics
        answer_metrics = evaluator.compute_answer_metrics(
            all_pred_answers, all_true_answers)
        explanation_metrics = evaluator.compute_explanation_metrics(
            predictions, references)

        return {**answer_metrics, **explanation_metrics}


class VQAExplanationGenerator:
    """
    A wrapper class for generating VQA explanations using the baseline model.
    """

    def __init__(self, model: VQABaselineModel):
        """
        Initialize the explanation generator.

        Args:
            model (VQABaselineModel): The baseline model to use for predictions
        """
        self.model = model

    def generate_explanation(self, image_path: str, question: str) -> Tuple[str, str]:
        """
        Generate an answer and explanation for a given image and question.

        Args:
            image_path (str): Path to the image
            question (str): Question text

        Returns:
            Tuple[str, str]: Generated answer and explanation
        """
        return self.model.predict(image_path, question)
