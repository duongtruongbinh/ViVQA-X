from typing import Callable, List
import importlib
import os
import re
from .data_processor import DataProcessor
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
            "ggtrans": ["googletrans===3.1.0a0", "httpx===0.13.3"]
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
                install_requirements(
                    self.required_packages[translator], translator)
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

        Raises:
            ValueError: If the file name does not contain 'train', 'val', or 'test'.
        """
        data = load_json(file_path)
        os.makedirs(self.translations_dir, exist_ok=True)
        file_name = os.path.basename(file_path)
        match = re.search(r"(train|val|test)", file_name)

        if not match:
            raise ValueError(
                f"File name {file_name} does not contain 'train', 'val', hoặc 'test'")

        file_type = match.group(0)
        translate_function = self.load_translator(translator)
        translated_data = translate_function(data, file_type)
        output_file = os.path.join(
            self.translations_dir, f"{os.path.splitext(file_name)[0]}_{translator}.json")
        save_json(translated_data, output_file)
        print(f"Translated data saved to {output_file}")

    def run(self):
        """
        Runs the translation phase for all dataset files and translators.
        """
        for dataset_file in self.dataset_files:
            for translator in self.translators:
                self.translate_and_save(dataset_file, translator)

        processor = DataProcessor(
            self.dataset_files, self.translations_dir, self.translators)
        processor.check_data_integrity()
        processor.merge_translations()
