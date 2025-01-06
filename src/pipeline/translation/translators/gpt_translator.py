import os
import json
import time
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from .base_translator import BaseTranslator
load_dotenv()


class GptTranslator(BaseTranslator):
    def __init__(self):
        """
        Initialize the GPTTranslator with required API key and dependencies.
        """
        super().__init__()
        self.requirements = ["openai", "httpx"]
        self.api_key = os.getenv("OPENAI_API_KEY")

    def prepare_batch_file(self, data: Dict[str, Dict], batch_file: str) -> None:
        """
        Prepare batch file for OpenAI processing.

        Args:
            data (Dict[str, Dict]): The dataset to be translated.
            batch_file (str): The path to the batch file to be created.
        """
        with open(batch_file, "w") as out_f:
            for key, value in data.items():
                common_answer = self.get_most_common_answer(value["answers"])
                messages = [
                    {
                        "role": "system",
                        "content": "You are a Vietnamese translator. Translate each section separated by '|' to Vietnamese. Provide only the translations, each on a new line, without any additional text or explanations.",
                    },
                    {
                        "role": "user",
                        "content": f"{value['question']}|{common_answer}|{'|'.join(value['explanation'])}",
                    },
                ]
                request = {
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": messages,
                        "max_tokens": 1000,
                    },
                }
                out_f.write(json.dumps(request) + "\n")

    def process_batch(self, batch_file: str) -> Tuple[Optional[str], str]:
        """
        Process batch through OpenAI API.

        Args:
            batch_file (str): The path to the batch file to be processed.

        Returns:
            Tuple[Optional[str], str]: The batch ID and status.
        """
        try:
            batch_input_file = openai.files.create(
                file=open(batch_file, "rb"), purpose="batch"
            )
            batch = openai.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "Translation job"},
            )

            while True:
                status = openai.batches.retrieve(batch.id).status
                print(f"\rBatch status: {status}", end="")
                if status in ["completed", "failed", "cancelled", "expired"]:
                    print()  # New line after status
                    return (batch.id, status) if status == "completed" else (None, status)
                time.sleep(60)

        except Exception as e:
            print(f"\nError in batch processing: {e}")
            return None, str(e)

    def download_results(self, batch_id: str) -> Optional[List[Dict]]:
        """
        Download and parse batch results.

        Args:
            batch_id (str): The ID of the batch to download results for.

        Returns:
            Optional[List[Dict]]: The list of results if successful, None otherwise.
        """
        try:
            batch_info = openai.batches.retrieve(batch_id)
            if not batch_info.output_file_id:
                print("No output file found")
                return None

            file_response = openai.files.content(batch_info.output_file_id)
            return [json.loads(line) for line in file_response.text.strip().split("\n")]

        except Exception as e:
            print(f"Error downloading results: {e}")
            return None

    def translate_batch(self, data: Dict[str, Dict], file_name: str) -> Dict[str, Dict]:
        """
        Main translation function that processes the entire dataset.

        Args:
            data (Dict[str, Dict]): The dataset to be translated.
            file_name (str): The name of the file being translated.

        Returns:
            Dict[str, Dict]: The translated dataset.
        """
        translated_dataset = {}
        print(f"Translating {file_name} with GPT-4o-mini")
        # Create temporary batch file
        batch_file = f"{file_name}_gpt_batch.jsonl"
        self.prepare_batch_file(data, batch_file)

        # Process batch
        batch_id, status = self.process_batch(batch_file)
        if not batch_id:
            print(f"Batch processing failed with status: {status}")
            return translated_dataset

        # Download and process results
        results = self.download_results(batch_id)
        if not results:
            print("Failed to download results")
            return translated_dataset

        # Parse results
        for result in tqdm(results, desc=f"Parsing translations for {file_name}"):
            try:
                key = result["custom_id"]
                content = result["response"]["body"]["choices"][0]["message"]["content"]

                if "\n" in content:
                    translations = [t.strip()
                                    for t in content.split("\n") if t.strip()]
                elif "|" in content:
                    translations = [t.strip()
                                    for t in content.split("|") if t.strip()]
                else:
                    print(f"Warning: Unexpected response format for {key}")
                    continue

                if len(translations) < 3:
                    print(f"Warning: Incomplete translation for {key}")
                    continue

                translated_dataset[key] = data[key].copy()
                translated_dataset[key].update(
                    {
                        "question_vi": translations[0],
                        "answer_vi": translations[1],
                        "explanation_vi": translations[2:],
                    }
                )

            except Exception as e:
                print(f"Error processing result for {key}: {e}")
                continue

        # Cleanup
        if os.path.exists(batch_file):
            os.remove(batch_file)

        return translated_dataset
