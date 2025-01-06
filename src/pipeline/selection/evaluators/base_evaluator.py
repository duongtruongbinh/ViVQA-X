from abc import ABC, abstractmethod
from typing import Dict, Any
from collections import Counter
import random
import torch

class BaseEvaluator(ABC):
    @abstractmethod
    def score(self, data: Dict[str, Any], file_name: str, translators: list[str]) -> list[Dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    def get_most_common_answer(self, answers):
        return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]
    
    def set_seed(self, seed: int = 0) -> None:
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