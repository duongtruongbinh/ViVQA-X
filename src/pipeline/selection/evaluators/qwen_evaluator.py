from .base_evaluator import BaseEvaluator
import torch
import regex as re
from transformers import pipeline
from typing import Dict, Any, List
from tqdm import tqdm

class QwenEvaluator(BaseEvaluator):
    def __init__(self):
        """Initialize the QwenEvaluator with a specific model."""
        self.model_name = "Qwen/Qwen2-7B-Instruct"
        self.model = None
        self.set_seed()
        
    def load_model(self) -> pipeline:
        """
        Load the text-generation model using the specified model name.

        Returns:
            pipeline: The loaded text-generation model pipeline.
        """
        return pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def create_evaluation_prompt(
        self,
        question: str,
        answer: str,
        explanation: str,
        eval_type: str,
        translations: List[str],
    ) -> str:
        """
        Create an evaluation prompt for the given question, answer, explanation, and translations.

        Args:
            question (str): The question in English.
            answer (str): The answer in English.
            explanation (str): The explanation in English.
            eval_type (str): The type of evaluation (question, answer, or explanation).
            translations (List[str]): The list of Vietnamese translations.

        Returns:
            str: The formatted evaluation prompt.
        """
        prompt = (
            f"""
You will evaluate {len(translations)} Vietnamese translations of the given {eval_type} from a Visual Question Answering (VQA) task. The context includes an English question, answer, and explanation. Your evaluation should be based on three criteria: accuracy, fluency, and cultural fit. 

Important:
- Do not judge the correctness of the answer itself, as it is based on the image and not the question.
- Ignore case differences when evaluating translations.
- Identical translations should receive the same score.
- Provide a score from 0 to 100 for each translation, one score per line, without any explanations.

Example:
Question: What is the capital of France?
Answer: The capital of France is Paris.
Explanation: Paris has been the capital of France since the Middle Ages.

Translations:
1: Thủ đô của nước Pháp là gì?
2: Thủ đô của Pháp là gì?

Output:
95
90

Now evaluate the following:

Question: {question}
Answer: {answer}
Explanation: {explanation}

Translations:
"""
            + "\n".join(f"{i+1}: {t}" for i, t in enumerate(translations))
            + """

Output:
"""
        )
        return prompt

    def score(self, data: Dict[str, Any], file_name: str, translators: list[str]) -> List[Dict[str, Any]]:
        """
        Score the translations for the given data.

        Args:
            data (Dict[str, Any]): The data containing questions, answers, and explanations.
            file_name (str): The name of the file being processed.
            translators (list[str]): The list of translators to evaluate.

        Returns:
            List[Dict[str, Any]]: The list of results with scores for each sample.
        """
        self.model = self.load_model()
        results = []
        for sample_id, sample in tqdm(data.items(), desc=f"Scoring {file_name} with Qwen"):
            common_answer = self.get_most_common_answer(sample["answers"])
            question_translations = [sample[f"question_vi_{translator}"] for translator in translators]
            answer_translations = [sample[f"answer_vi_{translator}"] for translator in translators]
            explanation_translations = [
                [sample[f"explanation_vi_{translator}"][i] for translator in translators]
                for i in range(len(sample["explanation"]))
            ]

            question_scores = self.evaluate_translations(
                sample["question"],
                common_answer,
                sample["explanation"][0],
                "question",
                question_translations,
            )
            answer_scores = self.evaluate_translations(
                sample["question"],
                common_answer,
                sample["explanation"][0],
                "answer",
                answer_translations,
            )
            explanation_scores = [
                self.evaluate_translations(
                    sample["question"],
                    common_answer,
                    sample["explanation"][i],
                    "explanation",
                    exp_translations,
                )
                for i, exp_translations in enumerate(explanation_translations)
            ]

            result = {
                "question_id": sample_id,
                "question": question_translations,
                "question_scores": question_scores,
                "answer": answer_translations,
                "answer_scores": answer_scores,
                "explanation": explanation_translations,
                "explanation_scores": explanation_scores,
            }

            results.append(result)
        return results

    def evaluate_translations(
        self,
        question: str,
        answer: str,
        explanation: str,
        eval_type: str,
        translations: List[str],
    ) -> List[int]:
        """
        Evaluate the given translations and return their scores.

        Args:
            question (str): The question in English.
            answer (str): The answer in English.
            explanation (str): The explanation in English.
            eval_type (str): The type of evaluation (question, answer, or explanation).
            translations (List[str]): The list of Vietnamese translations.

        Returns:
            List[int]: The list of scores for each translation.
        """
        prompt = self.create_evaluation_prompt(
            question, answer, explanation, eval_type, translations
        )
        try:
            with torch.no_grad():
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that can complete a given instruction.",
                    },
                    {"role": "user", "content": prompt},
                ]
                response = self.model(messages, **{
                    "max_new_tokens": 500,
                    "do_sample": True,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_return_sequences": 1,
                    "pad_token_id": 50256,
                })
                generated_text = response[0]["generated_text"]
                generated_text = generated_text[-1]["content"]
                scores = re.findall(r"\d+", generated_text)
                scores = [int(score.strip()) for score in scores]
                return scores
        except Exception as e:
            print(f"Error evaluating translations: {e}")
            return []

    @property
    def name(self) -> str:
        """
        Get the name of the evaluator.

        Returns:
            str: The name of the evaluator.
        """
        return "qwen"