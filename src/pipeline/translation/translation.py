# translation.py
import json
import os
import argparse
import regex as re
import subprocess
import importlib
from typing import Dict, Callable

# Define requirements for each translator
TRANSLATOR_REQUIREMENTS = {
    "vinai": [],
    "gemini": [],
    "ggtrans": ["googletrans===3.1.0a0", "httpx==0.13.3"],
    "gpt": ["openai"],
}


def uninstall_packages(packages: list):
    """Uninstall specified packages"""
    for package in packages:
        try:
            subprocess.check_call(
                ["pip", "uninstall", "-y", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Uninstalled {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error uninstalling {package}: {str(e)}")


def install_requirements(packages: list, source: str):
    """Install required packages if not already installed"""
    if source == "ggtrans":
        # Uninstall specific packages before installing ggtrans requirements
        uninstall_packages(["googletrans", "httpx"])

    for package in packages:
        try:
            importlib.import_module(package.split("==")[0])
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call(
                    ["pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {str(e)}")


def load_translator(source: str) -> Callable:
    """Dynamically import and return translator function"""
    if source not in TRANSLATOR_REQUIREMENTS:
        raise ValueError(f"Unknown translator source: {source}")

    # Install required packages
    install_requirements(TRANSLATOR_REQUIREMENTS[source], source)

    # Import translator module
    module = importlib.import_module(f"{source}_vqax")
    return getattr(module, "translate_batch")


def load_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate_and_save(file_path: str, output_dir: str, source: str):
    data = load_json(file_path)
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    match = re.search(r"(train|val|test)", file_name)
    if match:
        file_type = match.group(0)
    else:
        raise ValueError(
            f"File name {file_name} does not contain 'train', 'val', or 'test'"
        )
    try:
        translate_function = load_translator(source)
        translated_data = translate_function(data, file_type)

        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(file_name)[0]}_{source}.json",
        )
        save_json(translated_data, output_file)
        print(f"Translated data saved to {output_file}")

    except Exception as e:
        print(f"Error with translator '{source}': {str(e)}")


def check_data_integrity(
    dataset_files: list, output_dir: str, translation_sources: list
):
    def check_item_integrity(item, translated_item):
        errors = []
        question_key = f"question_vi"
        answer_key = f"answer_vi"
        explanation_key = f"explanation_vi"

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

    for dataset_file in dataset_files:
        print(f"Checking integrity for {dataset_file} dataset...")
        original_data = load_json(dataset_file)
        error_items = {}

        for source in translation_sources:
            translation_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(dataset_file))[0]}_{source}.json",
            )
            if not os.path.exists(translation_file):
                print(
                    f"Warning: {translation_file} not found. Skipping this translation source."
                )
                continue

            translated_data = load_json(translation_file)
            for key, item in original_data.items():
                if key not in translated_data:
                    error_items[key] = [f"Missing translation for key {key}"]
                else:
                    item_errors = check_item_integrity(item, translated_data[key])
                    if item_errors:
                        error_items[key] = item_errors

        # Save error items
        error_file = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(dataset_file))[0]}_errors.json",
        )
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_items, f, ensure_ascii=False, indent=2)

        print(f"Total items: {len(original_data)}")
        print(f"Items with errors: {len(error_items)}")
        print("\n")


def merge_translations(dataset_files: list, output_dir: str, translation_sources: list):
    for dataset_file in dataset_files:
        print(f"Processing {dataset_file} dataset...")
        original_data = load_json(dataset_file)
        translations = {}

        for source in translation_sources:
            translation_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(dataset_file))[0]}_{source}.json",
            )
            if os.path.exists(translation_file):
                translations[source] = load_json(translation_file)
            else:
                print(
                    f"Warning: {translation_file} not found. Skipping this translation source."
                )

        merged_data = {}
        for key, item in original_data.items():
            merged_item = item.copy()
            for source, translation in translations.items():
                if key in translation:
                    translated_item = translation[key]
                    merged_item[f"question_vi_{source}"] = translated_item[
                        "question_vi"
                    ]
                    merged_item[f"answer_vi_{source}"] = translated_item["answer_vi"]
                    merged_item[f"explanation_vi_{source}"] = translated_item[
                        "explanation_vi"
                    ]
            merged_data[key] = merged_item

        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(dataset_file))[0]}_translated.json",
        )
        save_json(merged_data, output_file)
        print(f"Merged data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate datasets using multiple translation models."
    )
    parser.add_argument(
        "dataset_files", nargs="+", help="Paths to the dataset files to be translated."
    )
    parser.add_argument(
        "--output_dir",
        default="../../../data/translation",
        help="Directory to save translated datasets.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["vinai", "gemini", "ggtrans", "gpt"],
        choices=list(TRANSLATOR_REQUIREMENTS.keys()),
        help="List of translation sources to use.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_file in args.dataset_files:
        for source in args.sources:
            translate_and_save(dataset_file, args.output_dir, source)

    check_data_integrity(
        args.dataset_files, args.output_dir, list(TRANSLATOR_REQUIREMENTS.keys())
    )
    merge_translations(
        args.dataset_files, args.output_dir, list(TRANSLATOR_REQUIREMENTS.keys())
    )
