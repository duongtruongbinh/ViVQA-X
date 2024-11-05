import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from collections import Counter
from typing import List, Dict


class PostProcessor:
    def __init__(
        self,
        dataset_dir: str,
        sampling_dir: str,
        output_dir: str,
        model_name: str,
        item_per_batch: int,
    ):
        self.dataset_dir = dataset_dir
        self.sampling_dir = sampling_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.item_per_batch = item_per_batch
        os.makedirs(output_dir, exist_ok=True)
        self.ner_ids_output_dir = os.path.join(output_dir, "ner_answers.json")

        # Set up CUDA environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        os.environ["WORLD_SIZE"] = "1"

        # Load model and tokenizer for NER
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Initialize NER pipeline with GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
            batch_size=item_per_batch,
        )

    def get_most_common_answer(self, answers: List[str]) -> str:
        return Counter(answers).most_common(1)[0][0]

    def contains_named_entity_in_answer(self, question: str, answer: str) -> List[Dict]:
        combined_text = f"{question} {answer}"
        ner_results = self.ner_pipeline(combined_text)
        if not ner_results:
            return []

        # Calculate answer position in combined_text
        answer_start = combined_text.find(answer)
        answer_end = answer_start + len(answer)

        entities = ["B-PER", "I-PER", "B-ORG", "I-ORG"]
        found_entities = []

        for ent in ner_results:
            if (
                ent["start"] >= answer_start
                and ent["end"] <= answer_end
                and ent["entity"] in entities
            ):
                found_entities.append(combined_text)

        return found_entities

    def process(self):
        ner_answers = []
        for dataset in ["train", "val", "test"]:
            # Load original and sampling data
            with open(os.path.join(self.dataset_dir, f"vqaX_{dataset}.json"), "r") as f:
                original_data = json.load(f)
            sampling_data = pd.read_csv(
                os.path.join(self.sampling_dir, f"{dataset}.csv")
            )

            for index in tqdm(range(len(sampling_data)), desc=f"Processing {dataset}"):
                sample = sampling_data.iloc[index]
                question_id = str(sample["question_id"])
                original_question = original_data[question_id]["question"]
                original_answer = self.get_most_common_answer(
                    original_data[question_id]["answers"]
                )
                img_id = original_data[question_id]["image_id"]

                # Check if there is a named entity specifically in the answer part
                ner_entities = self.contains_named_entity_in_answer(
                    original_question, original_answer
                )
                if ner_entities:
                    sampling_data.at[index, "answer"] = original_answer
                    ner_answers.extend(ner_entities)  # Save the details of NER entities

                # Add image ID to the sampling data
                sampling_data.at[index, "img_id"] = img_id

            # Remove columns
            sampling_data = sampling_data.drop(
                columns=[
                    "question_selection",
                    "answer_selection",
                    "explanation_selection",
                ]
            )
            sampling_data.to_csv(
                os.path.join(self.output_dir, f"vqaX_{dataset}_translated.csv"),
                index=False,
            )

        # Save the list of NER answers with combined text, entity type, and answer
        with open(self.ner_ids_output_dir, "w") as ner_file:
            json.dump(ner_answers, ner_file, ensure_ascii=False, indent=2)

        print(f"Saved NER answers to {self.ner_ids_output_dir}")


if __name__ == "__main__":
    dataset_dir = "../../../datasets/VQA-X"
    sampling_dir = os.path.join(dataset_dir, "study_case_add_llm/sampling_20")
    output_dir = os.path.join(dataset_dir, "final_data/post_processing")
    model_name = "dslim/bert-base-NER-uncased"
    item_per_batch = 5

    processor = PostProcessor(
        dataset_dir, sampling_dir, output_dir, model_name, item_per_batch
    )
    processor.process()
