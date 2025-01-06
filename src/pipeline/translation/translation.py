from typing import Callable, List, Dict
import importlib
import os
from .utils import load_json, save_json, install_requirements


class TranslationPhase:
    """
    A class to manage the translation phase of datasets using various translators.

    Methods:
        load_translator(translator: str) -> Callable: Dynamically imports and returns the translator function.
        translate_and_save(file_path: str, translator: str): Translates a dataset file and saves the translated data.
        run(): Runs the translation phase for all dataset files and translators.
    """

    def __init__(self, dataset_files: List[str], translations_dir: str, translators: List[str]):
        """
        Initializes the TranslationPhase with dataset files, output directory, and translators.

        Args:
            dataset_files (List[str]): List of dataset file paths.
            translations_dir (str): Directory to save translated files.
            translators (List[str]): List of translators to use.
        """
        self.dataset_files = dataset_files
        self.translations_dir = translations_dir
        self.translators = translators
        self.required_packages = {
            "gpt": ["openai", "httpx"],
            "ggtrans": ["googletrans===3.1.0a0"]
        }

    def load_translator(self, translator: str) -> Callable:
        """
        Dynamically imports and returns the translator function.

        Args:
            translator (str): The name of the translator.

        Returns:
            Callable: The translate_batch function of the translator.

        Raises:
            ValueError: If the translator module is not found
        """
        try:
            module_name = f"translation.translators.{translator}_translator"
            class_name = f"{translator.capitalize()}Translator"
            if translator in self.required_packages:
                install_requirements(self.required_packages[translator])
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise ValueError(f"Translator {translator} not found")

        translator_class = getattr(module, class_name)
        return translator_class().translate_batch

    def translate_and_save(self, file_path: str, translator: str):
        """
        Translates a dataset file and saves the translated data.

        Args:
            file_path (str): The path to the dataset file.
            translator (str): The name of the translator.
        """
        data = load_json(file_path)
        os.makedirs(self.translations_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        translate_function = self.load_translator(translator)
        translated_data = translate_function(data, file_name)

        output_file = os.path.join(self.translations_dir, f"{file_name}_{translator}.json")
        save_json(translated_data, output_file)
        print(f"Translated data saved to {output_file}")

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
            errors.append(f"Mismatch in number of explanations for {explanation_key}")
        elif any(not exp for exp in translated_item[explanation_key]):
            errors.append(f"Empty explanation in {explanation_key}")

        return errors

    def check_data_integrity(self) -> None:
        """
        Checks the integrity of translated data for all dataset files.
        """
        for dataset_file in self.dataset_files:
            print(f"Checking integrity for {dataset_file}")
            original_data = load_json(dataset_file)
            error_items = {}
            file_name = os.path.splitext(os.path.basename(dataset_file))[0]
            for translator in self.translators:
                translation_file = os.path.join(self.translations_dir, f"{file_name}_{translator}.json")
                if not os.path.exists(translation_file):
                    print(f"Warning: {translation_file} not found. Skipping this translation source.")
                    continue
                translated_data = load_json(translation_file)
                for key, item in original_data.items():
                    if key not in translated_data:
                        error_items[key] = [f"Missing translation for key {key}"]
                    else:
                        item_errors = self.check_item_integrity(item, translated_data[key])
                        if item_errors:
                            error_items[key] = item_errors
            error_file = os.path.join(self.translations_dir, f"{file_name}_errors.json")
            save_json(error_items, error_file)
            print(f"Total items: {len(original_data)} || Items with errors: {len(error_items)}\n")

    def merge_translations(self) -> None:
        """
        Merges translations from different translators for all dataset files.
        """
        for dataset_file in self.dataset_files:
            print(f"Processing {dataset_file} ...")
            original_data = load_json(dataset_file)

            # Load translations from all translators
            translations = {}
            file_name = os.path.splitext(os.path.basename(dataset_file))[0]
            for translator in self.translators:
                translation_file = os.path.join(self.translations_dir, f"{file_name}_{translator}.json")
                if os.path.exists(translation_file):
                    translations[translator] = load_json(translation_file)
                else:
                    print(f"Warning: {translation_file} not found. Skipping this translator.")

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
            output_file = os.path.join(self.translations_dir, f"{file_name}_translated.json")
            save_json(merged_data, output_file)
            print(f"Merged data saved to {output_file}")

    def run(self):
        """
        Runs the translation phase for all dataset files and translators.
        """
        print("Running translation phase ...")
        for translator in self.translators:
            for dataset_file in self.dataset_files:
                self.translate_and_save(dataset_file, translator)

        self.check_data_integrity()
        self.merge_translations()
