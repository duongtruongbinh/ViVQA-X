from .base_evaluator import BaseEvaluator
import os
import json
import time
import openai
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class GptEvaluator(BaseEvaluator):
    def __init__(self):
        """Initialize the GptEvaluator with specific settings."""
        self.batch_dir = os.path.join(os.path.dirname(__file__), "batches")
        os.makedirs(self.batch_dir, exist_ok=True)


    def generate_user_content(self, item: Dict[str, Any], source: List[str]) -> str:
        """
        Generate user content for the evaluation prompt.

        Args:
            item (Dict[str, Any]): The data item containing question, answer, and explanations.
            source (List[str]): The list of sources for translations.

        Returns:
            str: The generated user content.
        """
        question_translations = "\n".join(
            [
                f"Translation {idx + 1}: {item[f'question_vi_{source[idx]}']}"
                for idx in range(len(source))
            ]
        )
        answer_translations = "\n".join(
            [
                f"Translation {idx + 1}: {item[f'answer_vi_{source[idx]}']}"
                for idx in range(len(source))
            ]
        )

        explanation_translations = []
        for exp_idx, exp in enumerate(item["explanation"]):
            exp_translations = "\n".join(
                [
                    f"Translation {trans_idx + 1}: {item[f'explanation_vi_{source[trans_idx]}'][exp_idx]}"
                    for trans_idx in range(len(source))
                ]
            )
            explanation_translations.append(
                f"Explanation {exp_idx + 1}: {exp}\n{exp_translations}"
            )

        explanations_content = "\n\n".join(explanation_translations)

        user_content = f"""
        English Question: {item['question']}
        {question_translations}

        English Answer: {self.get_most_common_answer(item['answers'])}
        {answer_translations}

        {explanations_content}
        """
        return user_content.strip()

    def prepare_batch_input_files(self, data: Dict[str, Any], file_name: str, batch_size: int = 1500):
        """
        Prepare batch input files for OpenAI processing.

        Args:
            data (Dict[str, Any]): The dataset to be processed.
            file_name (str): The name of the dataset.
            batch_size (int): The size of each batch.
        """
        keys = list(data.keys())
        batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]

        for batch_idx, batch_keys in enumerate(batches):
            batch_output_file = os.path.join(self.batch_dir, f"{file_name}_batch_{batch_idx + 1}.jsonl")

            with open(batch_output_file, "w", encoding="utf-8") as out_f:
                for key in batch_keys:
                    value = data[key]
                    prompt = """You will be given an English question, answer, and explanations for context. Then, you will evaluate Vietnamese translations of the question, answer, and explanation(s). Evaluate each translation based on accuracy, fluency, and cultural appropriateness, considering the full context provided. Assign a score between 0 and 100 for each translation.
                    Different translations must have different scores. Ignore case differences. 
                    **Return the scores exactly in the following JSON format and no additional text or explanations:**
                    {{
                        "question_scores": [score_for_translation_1, score_for_translation_2, ...],
                        "answer_scores": [score_for_translation_1, score_for_translation_2, ...],
                        "explanation_scores": [
                            [scores for translations of explanation 1],
                            [scores for translations of explanation 2], (if has more than 1 explanation)
                            ...
                        ] (length of explanation_scores must match the number of explanations, example: n explanations -> n lists of scores in explanation_scores)
                    }}
                    Now, please evaluate the following:
                    """
                    user_content = self.generate_user_content(value, ["ggtrans", "gemini", "vinai", "gpt"])
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_content},
                    ]

                    request = {
                        "custom_id": key,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": messages,
                            "max_tokens": 200,
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"},
                        },
                    }

                    out_f.write(json.dumps(request) + "\n")

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to OpenAI.

        Args:
            file_path (str): The path to the file to be uploaded.

        Returns:
            str: The ID of the uploaded file.
        """
        try:
            with open(file_path, "rb") as f:
                batch_input_file = openai.files.create(file=f, purpose="batch")
            return batch_input_file.id
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def create_batch(self, input_file_id: str) -> str:
        """
        Create a batch for OpenAI processing.

        Args:
            input_file_id (str): The ID of the input file.

        Returns:
            str: The ID of the created batch.
        """
        try:
            batch = openai.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "Evaluation job"},
            )
            return batch.id
        except Exception as e:
            print(f"Error creating batch: {e}")
            return None

    def check_batch_status(self, batch_id: str) -> str:
        """
        Check the status of a batch.

        Args:
            batch_id (str): The ID of the batch.

        Returns:
            str: The status of the batch.
        """
        while True:
            batch_status = openai.batches.retrieve(batch_id)
            status = batch_status.status
            print(f"{batch_id}: {status}", end="\r")

            if status in ["completed", "failed", "cancelled", "expired"]:
                return status
            time.sleep(60)

    def download_batch_results(self, batch_id: str) -> str:
        """
        Download the results of a batch.

        Args:
            batch_id (str): The ID of the batch.

        Returns:
            str: The path to the downloaded results file.
        """
        try:
            batch_info = openai.batches.retrieve(batch_id)
            output_file_id = batch_info.output_file_id
            if not output_file_id:
                print("No result file found.")
                return None

            file_response = openai.files.content(output_file_id)

            output_file_path = os.path.join(self.batch_dir, f"{batch_id}_output.jsonl")
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(file_response.text)

            print(f"Results saved to {output_file_path}")
            return output_file_path
        except Exception as e:
            print(f"Error downloading batch results: {e}")
            return None

    def parse_batch_results(self, result_file: str, translated_data: Dict[str, Any], translators: List[str]) -> List[Dict[str, Any]]:
        """
        Parse the results of a batch.

        Args:
            result_file (str): The path to the result file.
            original_data (Dict[str, Any]): The original data.

        Returns:
            List[Dict[str, Any]]: The parsed results.
        """
        parsed_results = []

        with open(result_file, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f]

        for result in results:
            key = result["custom_id"]
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            scores = json.loads(content)

            if key in translated_data:
                item = translated_data[key]

                parsed_result = {
                    "question_id": key,
                    "question": [item[f"question_vi_{t}"] for t in translators],
                    "question_scores": scores["question_scores"],
                    "answer": [item[f"answer_vi_{t}"] for t in translators],
                    "answer_scores": scores["answer_scores"],
                    "explanation": [
                        [item[f"explanation_vi_{t}"][j] for t in translators]
                        for j in range(len(item["explanation"]))
                    ],
                    "explanation_scores": scores["explanation_scores"],
                }
                parsed_results.append(parsed_result)

        return parsed_results
        
    def score(self, data: Dict[str, Any], file_name: str, translators: List[str]) -> List[Dict[str, Any]]:
        """
        Score the translations for the given data.

        Args:
            data (Dict[str, Any]): The data containing questions, answers, and explanations.
            file_name (str): The name of the file being processed.

        Returns:
            List[Dict[str, Any]]: The list of results with scores for each sample.
        """
        print(f"Scoring {file_name} with GPT-4o-mini")
        self.prepare_batch_input_files(data, file_name)

        all_results = []

        for batch_file in os.listdir(self.batch_dir):
            if batch_file.startswith(f"{file_name}_batch") and batch_file.endswith(".jsonl"):
                batch_input_file_id = self.upload_file(os.path.join(self.batch_dir, batch_file))
                if not batch_input_file_id:
                    print("Error uploading batch input file.")
                    continue

                batch_id = self.create_batch(batch_input_file_id)
                if not batch_id:
                    print("Error creating batch.")
                    continue

                final_status = self.check_batch_status(batch_id)
                if final_status != "completed":
                    print(f"Batch processing failed with status: {final_status}")
                    continue

                result_file = self.download_batch_results(batch_id)
                if not result_file:
                    print("Error downloading batch results.")
                    continue

                batch_results = self.parse_batch_results(result_file, data, translators)
                all_results.extend(batch_results)

        # Clean up batch files
        for batch_file in os.listdir(self.batch_dir):
            os.remove(os.path.join(self.batch_dir, batch_file))
        os.rmdir(self.batch_dir)
        return all_results

    @property
    def name(self) -> str:
        """
        Get the name of the evaluator.

        Returns:
            str: The name of the evaluator.
        """
        return "gpt"