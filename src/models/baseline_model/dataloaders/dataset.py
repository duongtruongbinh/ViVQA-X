import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import Counter
from underthesea import word_tokenize
import json

class VQA_X_Dataset(Dataset):
    """
    VQADataset handles loading and preprocessing of VQA-X data, including images,
    questions, answers, and explanations.
    """

    def __init__(self,
                 data,
                 image_dir,
                 transform=None,
                 word2idx=None,
                 idx2word=None,
                 answer2idx=None,
                 idx2answer=None,
                 max_question_length=20,
                 max_explanation_length=15,
                 max_vocab_size=20000):
        """
        Initialize the VQA-X dataset.

        Args:
            data (dict): Dictionary containing the dataset.
            image_dir (str): Directory path where images are stored.
            transform (torchvision.transforms, optional): Transformations to apply to images.
            word2idx (dict, optional): Pre-built word to index mapping.
            idx2word (dict, optional): Pre-built index to word mapping.
            answer2idx (dict, optional): Pre-built answer to index mapping.
            idx2answer (dict, optional): Pre-built index to answer mapping.
            max_question_length (int): Maximum length for questions.
            max_explanation_length (int): Maximum length for explanations.
            max_vocab_size (int): Maximum size of the vocabulary.
        """
        # Normalize data to be a list of sample dicts
        if isinstance(data, dict):
            # Preserve stable order via keys to keep alignment across cached tensors
            self.data = [data[k] for k in data.keys()]
            self._data_is_dict = True
            self._data_keys = list(data.keys())
        elif isinstance(data, list):
            self.data = data
            self._data_is_dict = False
            self._data_keys = None
        else:
            raise TypeError(
                f"Unsupported data type {type(data)}. Expected list or dict of samples.")
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        # Use provided question_id when available; otherwise synthesize from index
        self.question_ids = [
            item.get('question_id', idx) for idx, item in enumerate(self.data)
        ]
        self.max_question_length = max_question_length
        self.max_explanation_length = max_explanation_length
        self.max_vocab_size = max_vocab_size

        # Initialize or set pre-built vocabularies
        if word2idx and idx2word and answer2idx and idx2answer:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.answer2idx = answer2idx
            self.idx2answer = idx2answer
        else:
            self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
            self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
            self.answer2idx = {}
            self.idx2answer = {}
            self.build_vocab()
            self.build_answer_vocab()

        # Preprocess and cache tokenized questions and explanations
        self.preprocessed_questions = []
        self.preprocessed_explanations = []
        self.preprocess_all()

    def build_vocab(self):
        """Build the word vocabulary from questions and explanations."""
        word_freq = Counter()
        # Iterate in the same order as question_ids to keep consistent caching
        for idx in range(len(self.data)):
            item = self.data[idx]
            question = item.get('question', '')
            word_freq.update(word_tokenize(question.lower()))

            explanations = item.get('explanation', [])
            if explanations and isinstance(explanations, list):
                word_freq.update(word_tokenize(explanations[0].lower()))
            else:
                explanation = explanations if isinstance(explanations, str) else ''
                word_freq.update(word_tokenize(explanation.lower()))

        # Add most common words to the vocabulary (-4 for special tokens)
        for word, _ in word_freq.most_common(self.max_vocab_size - 4):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def build_answer_vocab(self):
        """Build the answer vocabulary from answers."""
        answer_freq = Counter()
        for idx in range(len(self.data)):
            item = self.data[idx]
            # Support both formats: list of dicts in 'answers' or a single 'answer' string
            collected_answers = []
            if isinstance(item.get('answers', None), list):
                for ans in item.get('answers', []):
                    if isinstance(ans, dict):
                        a = ans.get('answer', '')
                        if isinstance(a, str) and len(a.strip()) > 0:
                            collected_answers.append(a.lower())
            # Fallback to single answer field
            single_answer = item.get('answer', '')
            if isinstance(single_answer, str) and len(single_answer.strip()) > 0:
                collected_answers.append(single_answer.lower())

            answer_freq.update(collected_answers)
            
        self.answer2idx = {'<UNK>': 0}
        self.idx2answer = {0: '<UNK>'}
        for answer, _ in answer_freq.most_common():
            if answer not in self.answer2idx:
                idx = len(self.answer2idx)
                self.answer2idx[answer] = idx
                self.idx2answer[idx] = answer

    def tokenize(self, text):
        """
        Tokenize and map words to indices.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of word indices.
        """
        tokens = word_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

    def pad_sequence(self, sequence, max_length):
        """
        Pad or truncate a sequence to a fixed length.

        Args:
            sequence (list): List of word indices.
            max_length (int): The desired sequence length.

        Returns:
            list: Padded or truncated sequence.
        """
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence + [self.word2idx['<PAD>']] * (max_length - len(sequence))

    def preprocess_all(self):
        """Preprocess and cache all questions and explanations."""
        for idx in range(len(self.data)):
            item = self.data[idx]
            # Preprocess question
            question = item.get('question', '')
            question_ids = self.tokenize(question)
            question_ids = self.pad_sequence(question_ids, self.max_question_length)
            self.preprocessed_questions.append(torch.LongTensor(question_ids))

            # Preprocess explanation
            explanations = item.get('explanation', [])
            if explanations and isinstance(explanations, list):
                explanation = explanations[0]
            elif isinstance(explanations, str):
                explanation = explanations
            else:
                explanation = ''

            explanation_ids = [self.word2idx['<START>']] + self.tokenize(explanation) + [self.word2idx['<END>']]
            explanation_ids = self.pad_sequence(explanation_ids, self.max_explanation_length)
            self.preprocessed_explanations.append(torch.LongTensor(explanation_ids))

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing image, question, answer, explanation, and question_id.
        """
        question_id = self.question_ids[idx]
        item = self.data[idx]

        # Load and process image
        image_path = os.path.join(self.image_dir, item.get('image_name', ''))
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {image_path}: {e}")

        image = self.transform(image)

        # Get preprocessed question and explanation
        question = self.preprocessed_questions[idx]
        explanation = self.preprocessed_explanations[idx]

        # Process answers
        answers_list = []
        if isinstance(item.get('answers', None), list):
            answers_list = [
                (ans.get('answer', '') if isinstance(ans, dict) else '').lower().strip()
                for ans in item['answers']
                if isinstance(ans, (dict, str))
            ]
        answers_list = [a for a in answers_list if a]

        if answers_list:
            most_common_answer = Counter(answers_list).most_common(1)[0][0]
            answer_idx = self.answer2idx.get(most_common_answer, self.answer2idx.get('<UNK>', 0))
        else:
            # Fallback: dùng 'answer' đơn
            single_answer = item.get('answer', '')
            if isinstance(single_answer, str):
                single_answer = single_answer.lower().strip()
            answer_idx = self.answer2idx.get(single_answer, self.answer2idx.get('<UNK>', 0))

        return {
            'image': image,
            'question': question,
            'answer': torch.tensor(answer_idx, dtype=torch.long),
            'explanation': explanation,
            'question_id': question_id
        }


def load_data(path):
    """
    Load JSON data from a given file path.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list: Loaded data as a list of sample dictionaries.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file {path} does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Accept multiple common formats and normalize to a list of sample dicts
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # Case 1: { "data": [ ... ] }
        if 'data' in raw and isinstance(raw['data'], list):
            return raw['data']
        # Case 2: { question_id: sample_dict, ... }
        if all(isinstance(v, dict) for v in raw.values()):
            return list(raw.values())
    raise ValueError(
        "Unsupported JSON structure. Expected a list of samples or a dict mapping ids to samples.")


def build_vocabularies(train_data, max_vocab_size=10537):
    """
    Build word and answer vocabularies from training data.

    Args:
        train_data (Union[list, dict]): Training dataset.
        max_vocab_size (int): Maximum vocabulary size.

    Returns:
        tuple: word2idx, idx2word, answer2idx, idx2answer dictionaries.
    """
    word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
    answer2idx = {'<UNK>': 0}
    idx2answer = {0: '<UNK>'}

    # Normalize iterable of items
    if isinstance(train_data, dict):
        items_iter = train_data.values()
    else:
        items_iter = train_data

    # Build word2idx and idx2word
    word_freq = Counter()
    for item in items_iter:
        question = item.get('question', '')
        word_freq.update(word_tokenize(question.lower()))

        explanations = item.get('explanation', [])
        if explanations and isinstance(explanations, list):
            word_freq.update(word_tokenize(explanations[0].lower()))
        elif isinstance(explanations, str):
            word_freq.update(word_tokenize(explanations.lower()))

    # -4 for special tokens
    for word, _ in word_freq.most_common(max_vocab_size - 4):
        if word not in word2idx:
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word

    # Build answer2idx and idx2answer
    if isinstance(train_data, dict):
        items_iter = train_data.values()
    else:
        items_iter = train_data

    answer_freq = Counter()
    for item in items_iter:
        if isinstance(item.get('answers', None), list):
            for ans in item['answers']:
                a = ans.get('answer', '') if isinstance(ans, dict) else str(ans)
                a = a.strip().lower()
                if a:
                    answer_freq.update([a])
        a_single = item.get('answer', '')
        if isinstance(a_single, str):
            a_single = a_single.strip().lower()
            if a_single:
                answer_freq.update([a_single])

    answer2idx = {'<UNK>': 0}
    idx2answer = {0: '<UNK>'}
    for answer, _ in answer_freq.most_common():
        if answer not in answer2idx:
            idx = len(answer2idx)
            answer2idx[answer] = idx
        idx2answer[idx] = answer

    return word2idx, idx2word, answer2idx, idx2answer