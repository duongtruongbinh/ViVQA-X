import random
import torch
from collections import Counter
from typing import List, Dict, Any


def set_seed(seed: int = 0) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_most_common_answer(answers: List[Dict[str, Any]]) -> str:
    """
    Get the most common answer from a list of answers.

    Args:
        answers (List[Dict[str, Any]]): List of answer dictionaries.

    Returns:
        str: The most common answer.
    """
    return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]
