import os
import json
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
import torch
from collections import Counter


class PostProcessingPhase:
    """
    Class to manage the post-processing phase of the pipeline.
    
    Methods:
        get_most_common_answer(answers: List[Dict[str, Any]]) -> str: Get the most common answer from a list of answers.
        contains_named_entity_in_answer(self, question: str, answer: str) -> List[str]: Check if the answer contains named entities.
        run(): Run the post-processing phase.
    """
    def __init__(self, sampling_dir: str, output_dir: str, original_files: List[str], model_name: str):
        """
        Initialize the PostProcessingPhase.

        Args:
            sampling_dir (str): Directory containing the selection phase output.
            output_dir (str): Directory to save the final output files.
            original_files (List[str]): List of paths to the original files.
            model_name (str): Name of the model to use for post-processing.
        """
        self.sampling_dir = sampling_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.original_files = original_files
        self.item_per_batch = 5

    def get_most_common_answer(self, answers: List[Dict[str, Any]]) -> str:
        """
        Get the most common answer from a list of answers.
        """
        return Counter(answer['answer'] for answer in answers).most_common(1)[0][0]
    def contains_named_entity_in_answer(self, question: str, answer: str) -> List[str]:
        """
        Check if the answer contains named entities.

        Args:
            question (str): The question text.
            answer (str): The answer text.

        Returns:
            List[str]: List of found named entities.
        """
        combined_text = f"{question} {answer}"
        ner_results = self.ner_pipeline(combined_text)

        # Calculate answer position in combined_text
        answer_start = combined_text.find(answer)
        answer_end = answer_start + len(answer)
        
        entities = ["B-PER", "I-PER", "B-ORG", "I-ORG"]
        found_entities = []
        
        for ent in ner_results:
            # Check if the entity is within the answer part
            if ent["start"] >= answer_start and ent["end"] <= answer_end and ent["entity"] in entities:
                found_entities.append(combined_text)
    
        
        return found_entities

    def run(self):
        """
        Run the post-processing phase.
        """
        print("Running post-processing phase...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0, batch_size=self.item_per_batch)
        
        for file in self.original_files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            with open(file, "r") as f:
                original_data = json.load(f)
            with open(os.path.join(self.sampling_dir, f"{file_name}_sampling_voting.json"), "r") as f:
                sampling_data = json.load(f)
            output_data = sampling_data.copy()
            
            for question_id, data in tqdm(original_data.items(), desc=f"Processing {file_name}"):
                original_question = data["question"]
                original_answer = self.get_most_common_answer(data["answers"])

                ner_entities = self.contains_named_entity_in_answer(original_question, original_answer)
                if ner_entities:
                    output_data[question_id]["answer"] = original_answer
            output_file_name = file_name.replace("vqaX", "ViVQA-X")
            output_file_path = os.path.join(self.output_dir, f"{output_file_name}.json")
            with open(output_file_path, "w") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Saved post-processed file to {output_file_path}")
 