import json
from typing import Any, Dict
import numpy as np
import random

def softmax(arr: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of an array.
    
    Args:
        arr (np.ndarray): Input array.
    
    Returns:
        np.ndarray: Softmax of the input array.
    """
    e_x = np.exp(arr - np.max(arr))
    return e_x / np.sum(e_x)

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Loads a JSON file from the given file path.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        Dict[str, Any]: Loaded JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str) -> None:
    """
    Saves data to a JSON file at the given file path.
    
    Args:
        data (Any): Data to be saved.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
def set_seed(seed: int = 0) -> None:
    """
    Set the seed for reproducibility.
    
    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
