import os
import json
from typing import List, Dict
from .utils import load_json, save_json


class DataProcessor:
    """
    A class to handle data processing tasks including integrity checking and merging translations.

    Attributes:
        dataset_files (List[str]): List of dataset file paths.
        output_dir (str): Directory to save processed files.
        translators (List[str]): List of translation sources.
    """

    def __init__(self, dataset_files: List[str], output_dir: str, translators: List[str]):
        """
        Initializes the DataProcessor with dataset files, output directory, and translation sources.

        Args:
            dataset_files (List[str]): List of dataset file paths.
            output_dir (str): Directory to save processed files.
            translators (List[str]): List of translation sources.
        """
        self.dataset_files = dataset_files
        self.output_dir = output_dir
        self.translators = translators

    def check_item_integrity(self, item: Dict, translated_item: Dict) -> List[str]:
        """
        Checks the integrity of a single translated item.

        Args:
            item (Dict): The original item.
            translated_item (Dict): The translated item.

        Returns:
            List[str]: A list of error messages, if any.
        """
        errors = []
        question_key = "question_vi"
        answer_key = "answer_vi"
        explanation_key = "explanation_vi"

        if not translated_item.get(question_key):
            errors.append(f"Missing {question_key}")
        if not translated_item.get(answer_key):
            errors.append(f"Missing {answer_key}")
        if explanation_key not in translated_item:
            errors.append(f"Missing {explanation_key}")
        elif not translated_item[explanation_key]:
            errors.append(f"Empty {explanation_key}")
        elif len(translated_item[explanation_key]) != len(item["explanation"]):
            errors.append(
                f"Mismatch in number of explanations for {explanation_key}")
        elif any(not exp for exp in translated_item[explanation_key]):
            errors.append(f"Empty explanation in {explanation_key}")

        return errors

    def check_data_integrity(self) -> None:
        """
        Checks the integrity of translated data for all dataset files.
        """
        for dataset_file in self.dataset_files:
            print(f"Checking integrity for {dataset_file} dataset...")
            original_data = load_json(dataset_file)
            error_items = {}
            file_name = os.path.splitext(os.path.basename(dataset_file))[0]
            for translator in self.translators:
                translation_file = os.path.join(
                    self.output_dir, f"{file_name}_{translator}.json"
                )
                if not os.path.exists(translation_file):
                    print(
                        f"Warning: {translation_file} not found. Skipping this translation source.")
                    continue
                translated_data = load_json(translation_file)
                for key, item in original_data.items():
                    if key not in translated_data:
                        error_items[key] = [
                            f"Missing translation for key {key}"]
                    else:
                        item_errors = self.check_item_integrity(
                            item, translated_data[key])
                        if item_errors:
                            error_items[key] = item_errors
            error_file = os.path.join(
                self.output_dir, f"{file_name}_errors.json"
            )
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_items, f, ensure_ascii=False, indent=2)
            print(f"Total items: {len(original_data)}")
            print(f"Items with errors: {len(error_items)}\n")

    def merge_translations(self) -> None:
        """
        Merges translations from different translators for all dataset files.
        """
        for dataset_file in self.dataset_files:
            print(f"Processing {dataset_file} dataset...")
            original_data = load_json(dataset_file)

            # Load translations from all translators
            translations = {}
            file_name = os.path.splitext(os.path.basename(dataset_file))[0]
            for translator in self.translators:
                translation_file = os.path.join(
                    self.output_dir, f"{file_name}_{translator}.json")
                if os.path.exists(translation_file):
                    translations[translator] = load_json(translation_file)
                else:
                    print(
                        f"Warning: {translation_file} not found. Skipping this translator.")

            # Merge translations
            merged_data = {}
            for key, item in original_data.items():
                merged_item = item.copy()
                for translator, translation in translations.items():
                    if key in translation:
                        translated_item = translation[key]
                        merged_item[f"question_vi_{translator}"] = translated_item["question_vi"]
                        merged_item[f"answer_vi_{translator}"] = translated_item["answer_vi"]
                        merged_item[f"explanation_vi_{translator}"] = translated_item["explanation_vi"]
                merged_data[key] = merged_item
            output_file = os.path.join(
                self.output_dir, f"{file_name}_translated.json")
            save_json(merged_data, output_file)
            print(f"Merged data saved to {output_file}")
