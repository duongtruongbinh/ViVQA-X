import os
import json
from typing import List, Dict, Any
from .evaluators.base_evaluator import BaseEvaluator
from .evaluators.llama_evaluator import LlamaEvaluator
from .evaluators.gemma_evaluator import GemmaEvaluator
from .evaluators.phi_evaluator import PhiEvaluator
from .evaluators.qwen_evaluator import QwenEvaluator
from .evaluators.gpt_evaluator import GptEvaluator

evaluators_dict = {
    "llama": LlamaEvaluator(),
    "gemma": GemmaEvaluator(),
    "phi": PhiEvaluator(),
    "qwen": QwenEvaluator(),
    "gpt": GptEvaluator()
}

class SelectionPhase:
    def __init__(self, evaluators: List[str], translators: List[str], translations_dir: str, selection_dir: str):
        self.evaluators = [evaluators_dict[evaluator] for evaluator in evaluators]
        self.translators = translators
        self.translations_dir = translations_dir
        self.selection_dir = selection_dir
        self.scoring_dir = os.path.join(selection_dir, 'scoring')
        self.sampling_dir = os.path.join(selection_dir, 'sampling')
        self.voting_dir = os.path.join(selection_dir, 'voting')
        self.translated_files = [f for f in os.listdir(self.translations_dir) if f.endswith('_translated.json')]

    def run_scoring(self):
        print("Running scoring step ...")
        evaluation_files = set()
        for evaluator in self.evaluators:
            for file in self.translated_files:
                data = self.load_json(os.path.join(self.translations_dir, file))
                file_name = file.replace('_translated.json', '')
                evaluation_files.add(file_name)
                output_name = f"{file_name}_{evaluator.name}_evaluation.json"
                output_path = os.path.join(self.scoring_dir, output_name)
                
                scoring_data = evaluator.score(data, file_name, self.translators)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(scoring_data, f, indent=2, ensure_ascii=False)
                    
                print(f"Scoring for {file} with {evaluator.name} saved to {output_path}")

        self.check_and_merge_scores(list(evaluation_files))

    def check_and_merge_scores(self, evaluation_files: List[str]):
        llm_models = [evaluator.name for evaluator in self.evaluators]

        # Check for errors in the evaluation files
        print("Checking for errors in evaluation files ...")
        for model in llm_models:
            for eval_file in evaluation_files:
                file_path = os.path.join(self.scoring_dir, f"{eval_file}_{model}_evaluation.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        errors = self.check_scores_length(data)
                        if errors:
                            print(f"Errors in {file_path}:")
                            for error in errors:
                                print(f"  - {error}")
                        else:
                            print(f"No errors found in {eval_file}_{model}_evaluation.json")
                else:
                    print(f"File {file_path} does not exist.")

        # Merge the evaluation files
        print("Merging evaluation files ...")
        for eval_file in evaluation_files:
            json_files = {
                model: os.path.join(self.scoring_dir, f"{eval_file}_{model}_evaluation.json")
                for model in llm_models
            }
            merged_data = self.merge_evaluation_data(**json_files)
            output_path = os.path.join(self.scoring_dir, f'{eval_file}_evaluation.json')
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(merged_data, outfile, indent=2, ensure_ascii=False)
            print(f"Merged data for {eval_file} saved to {output_path}")

    def check_scores_length(self, data):
        errors = []
        for entry in data:
            if len(entry['question_scores']) != len(entry['question']):
                errors.append(f"Mismatch in question_scores: {entry['question_id']}")
            if len(entry['answer_scores']) != len(entry['answer']):
                errors.append(f"Mismatch in answer_scores: {entry['question_id']}")
            for i, expl_scores in enumerate(entry['explanation_scores']):
                try:
                    if len(expl_scores) != len(entry['explanation'][i]):
                        errors.append(f"Mismatch in explanation_scores[{i}]: {entry['question_id']}")
                except:
                    errors.append(f"Error in explanation_scores[{i}]: {entry['question_id']}")
        return errors

    def merge_evaluation_data(self, **json_files):
        merged_data = {}
        for model_name, file_path in json_files.items():
            data = self.load_json(file_path)
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

    def load_json(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_sampling(self):
        pass

    def run_voting(self):
        pass

    def run(self):
        print("Running selection phase ...")
        
        self.run_scoring()
        self.run_sampling()
        self.run_voting()
