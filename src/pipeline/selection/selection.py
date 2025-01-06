import os
import numpy as np
from collections import Counter
from typing import List, Dict, Any
from .evaluators.llama_evaluator import LlamaEvaluator
from .evaluators.gemma_evaluator import GemmaEvaluator
from .evaluators.phi_evaluator import PhiEvaluator
from .evaluators.qwen_evaluator import QwenEvaluator
from .evaluators.gpt_evaluator import GptEvaluator
from .utils import load_json, save_json, softmax, set_seed
from tqdm import tqdm


evaluators_dict = {
    "llama": LlamaEvaluator(),
    "gemma": GemmaEvaluator(),
    "phi": PhiEvaluator(),
    "qwen": QwenEvaluator(),
    "gpt": GptEvaluator()
}

avr_scores = {
    "gpt": 0.6287,
    "llama": 0.4767,
    "gemma": 0.5787,
    "qwen": 0.5234,
    "phi": 0.4356
}


class SelectionPhase:
    """
    Class to manage the selection phase of datasets using various evaluators.

    Methods:
        run(): Runs the entire selection phase.
        run_scoring(): Runs the scoring step for all evaluators and translated files.
        check_and_merge_scores(evaluation_files: List[str]): Checks for errors in the evaluation files and merges them.
        check_scores_length(data: List[Dict[str, Any]]): Checks the length of scores in the evaluation data.
        merge_evaluation_data(**json_files: str): Merges evaluation data from multiple JSON files.
        run_sampling(): Placeholder for the sampling step.
        run_voting(): Placeholder for the voting step.
    """

    def __init__(self, evaluators: List[str], translators: List[str], translations_dir: str, selection_dir: str):
        """
        Initializes the SelectionPhase with evaluators, translators, directories, and translated files.

        Args:
            evaluators (List[str]): List of evaluator names.
            translators (List[str]): List of translator names.
            translations_dir (str): Directory containing translation files.
            selection_dir (str): Directory for selection phase outputs.
        """
        self.evaluators = [evaluators_dict[evaluator]
                           for evaluator in evaluators]
        self.translators = translators
        self.translations_dir = translations_dir
        self.selection_dir = selection_dir
        self.scoring_dir = os.path.join(selection_dir, 'scoring')
        self.sampling_dir = os.path.join(selection_dir, 'sampling_voting')
        self.voting_dir = os.path.join(selection_dir, 'voting')
        self.translated_files = [f for f in os.listdir(
            self.translations_dir) if f.endswith('_translated.json')]

    def run_scoring(self) -> None:
        """
        Runs the scoring step for all evaluators and translated files.
        """
        print("Running scoring step ...")
        evaluation_files = set()
        for evaluator in self.evaluators:
            for file in self.translated_files:
                data = load_json(os.path.join(self.translations_dir, file))
                file_name = file.replace('_translated.json', '')
                evaluation_files.add(file_name)
                output_name = f"{file_name}_{evaluator.name}_evaluation.json"
                output_path = os.path.join(self.scoring_dir, output_name)

                scoring_data = evaluator.score(
                    data, file_name, self.translators)
                save_json(scoring_data, output_path)

                print(
                    f"Scoring for {file} with {evaluator.name} saved to {output_path}")

        self.check_and_merge_scores(list(evaluation_files))

    def check_and_merge_scores(self, evaluation_files: List[str]) -> None:
        """
        Checks for errors in the evaluation files and merges them.

        Args:
            evaluation_files (List[str]): List of evaluation file names.
        """
        llm_models = [evaluator.name for evaluator in self.evaluators]

        print("Checking for errors in evaluation files ...")
        for model in llm_models:
            for eval_file in evaluation_files:
                file_path = os.path.join(
                    self.scoring_dir, f"{eval_file}_{model}_evaluation.json")
                if os.path.exists(file_path):
                    data = load_json(file_path)
                    errors = self.check_scores_length(data)
                    if errors:
                        print(f"Errors in {file_path}:")
                        for error in errors:
                            print(f"  - {error}")
                    else:
                        print(
                            f"No errors found in {eval_file}_{model}_evaluation.json")
                else:
                    print(f"File {file_path} does not exist.")

        print("Merging evaluation files ...")
        for eval_file in evaluation_files:
            json_files = {
                model: os.path.join(
                    self.scoring_dir, f"{eval_file}_{model}_evaluation.json")
                for model in llm_models
            }
            merged_data = self.merge_evaluation_data(**json_files)
            output_path = os.path.join(
                self.scoring_dir, f'{eval_file}_evaluation.json')
            save_json(merged_data, output_path)
            print(f"Merged data for {eval_file} saved to {output_path}")

    def check_scores_length(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Checks the length of scores in the evaluation data.

        Args:
            data (List[Dict[str, Any]]): List of evaluation data entries.

        Returns:
            List[str]: List of error messages.
        """
        errors = []
        for entry in data:
            if len(entry['question_scores']) != len(entry['question']):
                errors.append(
                    f"Mismatch in question_scores: {entry['question_id']}")
            if len(entry['answer_scores']) != len(entry['answer']):
                errors.append(
                    f"Mismatch in answer_scores: {entry['question_id']}")
            for i, expl_scores in enumerate(entry['explanation_scores']):
                try:
                    if len(expl_scores) != len(entry['explanation'][i]):
                        errors.append(
                            f"Mismatch in explanation_scores[{i}]: {entry['question_id']}")
                except:
                    errors.append(
                        f"Error in explanation_scores[{i}]: {entry['question_id']}")
        return errors

    def merge_evaluation_data(self, **json_files: str) -> List[Dict[str, Any]]:
        """
        Merges evaluation data from multiple JSON files.

        Args:
            **json_files (str): Dictionary of model names to file paths.

        Returns:
            List[Dict[str, Any]]: Merged evaluation data.
        """
        merged_data = {}
        for model_name, file_path in json_files.items():
            data = load_json(file_path)
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
                        "explanation_scores": {model_name: entry["explanation_scores"]}
                    }
                merged_data[question_id]["question_scores"][model_name] = entry["question_scores"]
                merged_data[question_id]["answer_scores"][model_name] = entry["answer_scores"]
                merged_data[question_id]["explanation_scores"][model_name] = entry["explanation_scores"]
        return list(merged_data.values())

    def run_sampling(self, scores: List[int], temperature: float, n_samples: int = 20, temperature_scaling: bool = True) -> int:
        """
        Samples indices based on the provided scores and temperature.

        Args:
            scores (List[int]): The scores to sample from.
            temperature (float): The temperature for scaling the scores.
            n_samples (int, optional): The number of samples to draw. Defaults to 20.
            temperature_scaling (bool, optional): Whether to apply temperature scaling. Defaults to True.

        Returns:
            int: The most common sampled index.
        """
        if temperature_scaling:
            softmax_scores = softmax(np.array(scores) / temperature)
        else:
            softmax_scores = softmax(np.array(scores))
        samples = [np.random.choice(len(scores), p=softmax_scores)
                   for _ in range(n_samples)]
        most_common = Counter(samples).most_common(1)[0][0]
        return most_common

    def run_voting(self) -> None:
        """
        Runs the voting process on the translated files.
        """
        for file in self.translated_files:
            scoring_file = file.replace('_translated.json', '_evaluation.json')
            scoring_data = load_json(os.path.join(
                self.scoring_dir, scoring_file))
            translation_data = load_json(
                os.path.join(self.translations_dir, file))
            chosen_indices = {"question": None,
                              "answer": None, "explanation": []}
            result_data = {}
            temperature_score = {key: round(1 - value, 2)
                                 for key, value in avr_scores.items()}

            def get_highest_voted_index_with_tiebreak(votes: List[int], store_avg_scores: List[float]) -> int:
                """
                Gets the index with the highest votes, breaking ties with average scores.

                Args:
                    votes (List[int]): The list of votes.
                    store_avg_scores (List[float]): The list of average scores.

                Returns:
                    int: The index with the highest votes or highest average score in case of a tie.
                """
                max_votes = max(votes)
                candidates = [index for index, vote in enumerate(
                    votes) if vote == max_votes]

                if len(candidates) == 1:
                    return candidates[0]
                scores = [store_avg_scores[i] for i in candidates]
                return candidates[scores.index(max(scores))]

            for i, sample_input in tqdm(enumerate(scoring_data), total=len(scoring_data), desc=f"Sampling & Voting for {file}"):
                id = sample_input["question_id"]
                for key in ["question", "answer"]:
                    store_votes = [0] * len(sample_input[key])
                    store_avg_scores = [0] * len(sample_input[key])
                    model_scores = sample_input[f"{key}_scores"]

                    for model, scores in model_scores.items():
                        normalized_scores = [s/100 for s in scores]
                        chosen_index = self.run_sampling(
                            normalized_scores, temperature_score[model], n_samples=20, temperature_scaling=True)
                        store_votes[chosen_index] += 1
                        store_avg_scores[chosen_index] += avr_scores[model]

                    chosen_indices[key] = get_highest_voted_index_with_tiebreak(
                        store_votes, store_avg_scores)

                for i, explanations in enumerate(sample_input["explanation"]):
                    store_votes_explanation = [0] * len(explanations)
                    store_avg_scores_explanation = [0] * len(explanations)
                    model_scores_explanation = {
                        model: [s/100 for s in sample_input["explanation_scores"][model][i]] for model in sample_input["explanation_scores"]
                    }
                    for model, scores in model_scores_explanation.items():
                        chosen_index = self.run_sampling(
                            scores, temperature_score[model], n_samples=20, temperature_scaling=True)
                        store_votes_explanation[chosen_index] += 1
                        store_avg_scores_explanation[chosen_index] += avr_scores[model]
                    chosen_indices["explanation"].append(
                        get_highest_voted_index_with_tiebreak(
                            store_votes_explanation, store_avg_scores_explanation)
                    )

                result_data[id] = {
                    "question": sample_input["question"][chosen_indices["question"]],
                    "answer": sample_input["answer"][chosen_indices["answer"]],
                    "image_id": translation_data[id]["image_id"],
                    "image_name": translation_data[id]["image_name"],
                    "explanation": [explanations[chosen_indices["explanation"][i]] for i, explanations in enumerate(sample_input["explanation"])]
                }

            output_path = os.path.join(self.sampling_dir, file.replace(
                '_translated.json', '_sampling_voting.json'))
            save_json(result_data, output_path)
            print(f"Voting results for {file} saved to {output_path}")

    def run(self) -> None:
        """
        Runs the entire selection phase.
        """
        print("Running selection phase ...")
        set_seed()
        self.run_scoring()
        self.run_voting()
