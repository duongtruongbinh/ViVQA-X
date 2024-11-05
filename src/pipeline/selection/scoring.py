import argparse
import logging
from pathlib import Path
import json
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from gpt_evaluate import process_dataset as gpt_evaluate
from evaluate_translations import main as llm_evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScoringCoordinator:
    SUPPORTED_MODELS = ["llama", "gemma", "phi", "qwen"]
    SUPPORTED_DATASETS = ["train", "val", "test"]

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir
        self.vqax_dir = os.path.join(self.base_dir, "VQA-X")
        self.evaluation_dir = os.path.join(self.vqax_dir, "evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def run_gpt_evaluation(self, dataset: str) -> None:
        """Run GPT-4 evaluation for a specific dataset"""
        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset} not supported")

        logger.info(f"Starting GPT evaluation for dataset: {dataset}")
        try:
            gpt_evaluate(dataset)
            logger.info(f"Completed GPT evaluation for dataset: {dataset}")
        except Exception as e:
            logger.error(f"Error in GPT evaluation: {e}")
            raise

    def run_llm_evaluation(self, model: str) -> None:
        """Run specific LLM evaluation across all datasets"""
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not supported")

        logger.info(f"Starting evaluation with model: {model}")
        try:
            llm_evaluate(model)
            logger.info(f"Completed evaluation with model: {model}")
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            raise

    def run_all_evaluations(self, datasets: List[str], models: List[str]) -> None:
        """Run all specified evaluations in parallel"""
        logger.info("Starting comprehensive evaluation")

        with ThreadPoolExecutor() as executor:
            # Submit GPT evaluations
            gpt_futures = [
                executor.submit(self.run_gpt_evaluation, dataset)
                for dataset in datasets
            ]

            # Submit LLM evaluations
            llm_futures = [
                executor.submit(self.run_llm_evaluation, model) for model in models
            ]

            # Wait for all evaluations to complete
            for future in gpt_futures + llm_futures:
                future.result()

    def aggregate_results(self, dataset: str) -> dict:
        """Aggregate results from all evaluations for a dataset"""
        results = {
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "evaluations": {},
        }

        # Collect GPT results
        gpt_file = os.path.join(self.evaluation_dir, f"{dataset}_gpt_evaluation.json")
        if os.path.exists(gpt_file):
            with open(gpt_file, "r", encoding="utf-8") as f:
                results["evaluations"]["gpt"] = json.load(f)

        # Collect LLM results
        for model in self.SUPPORTED_MODELS:
            llm_file = os.path.join(
                self.evaluation_dir, f"{dataset}_{model}_evaluation.json"
            )
            if os.path.exists(llm_file):
                with open(llm_file, "r", encoding="utf-8") as f:
                    results["evaluations"][model] = json.load(f)

        return results

    def check_scores_length(self, data):
        """Check the length of scores in the evaluation data"""
        errors = []
        for entry in data:
            if len(entry["question_scores"]) != len(entry["question"]):
                errors.append(f"Mismatch in question_scores: {entry['question_id']}")
            if len(entry["answer_scores"]) != len(entry["answer"]):
                errors.append(f"Mismatch in answer_scores: {entry['question_id']}")
            for i, expl_scores in enumerate(entry["explanation_scores"]):
                try:
                    if len(expl_scores) != len(entry["explanation"][i]):
                        errors.append(
                            f"Mismatch in explanation_scores[{i}]: {entry['question_id']}"
                        )
                except:
                    errors.append(
                        f"Error in explanation_scores[{i}]: {entry['question_id']}"
                    )
        return errors

    def merge_evaluation_data(self, dataset_name):
        """Merge evaluation data from all models into a single file"""
        merged_data = {}
        for model in self.SUPPORTED_MODELS:
            file_path = os.path.join(
                self.evaluation_dir, f"{dataset_name}_{model}_evaluation.json"
            )
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    errors = self.check_scores_length(data)
                    if errors:
                        logger.error(f"Errors in {file_path}:")
                        for error in errors:
                            logger.error(f"  - {error}")
                        continue
                    for entry in data:
                        question_id = entry["question_id"]
                        if question_id not in merged_data:
                            merged_data[question_id] = {
                                "question_id": question_id,
                                "question": entry["question"],
                                "question_scores": {},
                                "answer": entry["answer"],
                                "answer_scores": {},
                                "explanation": entry["explanation"],
                                "explanation_scores": {},
                            }
                        merged_data[question_id]["question_scores"][model] = entry[
                            "question_scores"
                        ]
                        merged_data[question_id]["answer_scores"][model] = entry[
                            "answer_scores"
                        ]
                        merged_data[question_id]["explanation_scores"][model] = entry[
                            "explanation_scores"
                        ]
            else:
                logger.warning(f"File {file_path} does not exist.")

        output_file = os.path.join(
            self.evaluation_dir, f"vqaX_{dataset_name}_evaluation.json"
        )
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(list(merged_data.values()), outfile, indent=2, ensure_ascii=False)
        logger.info(f"Merged data for {dataset_name} saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Coordinate translation evaluation process"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ScoringCoordinator.SUPPORTED_DATASETS,
        default=["train", "val", "test"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ScoringCoordinator.SUPPORTED_MODELS,
        default=["llama", "gemma", "phi", "qwen"],
        help="Models to use for evaluation",
    )

    args = parser.parse_args()
    coordinator = ScoringCoordinator()

    try:
        for dataset in args.datasets:
            results = coordinator.aggregate_results(dataset)
            output_file = os.path.join(
                coordinator.evaluation_dir, f"{dataset}_aggregated_results.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Aggregated results saved to {output_file}")

        for dataset in args.datasets:
            coordinator.merge_evaluation_data(dataset)

    except Exception as e:
        logger.error(f"Error during evaluation process: {e}")
        raise


if __name__ == "__main__":
    main()
