from abc import ABC, abstractmethod
from typing import Dict


class BaseTranslator(ABC):
    @abstractmethod
    def translate_batch(self, data: Dict, file_type: str) -> Dict:
        pass
