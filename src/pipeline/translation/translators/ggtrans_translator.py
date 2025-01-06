from googletrans import Translator
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time
import random
from .base_translator import BaseTranslator


class GgtransTranslator(BaseTranslator):
    def __init__(self):
        """
        Initialize the GGTransTranslator with required dependencies.
        """
        super().__init__()
        self.requirements = ["googletrans===3.1.0a0", "httpx==0.13.3"]

    def safe_translate(self, translator: Translator, text: str, src: str = "en", dest: str = "vi", max_retries: int = 3) -> str:
        """
        Safely translate text with retries.

        Args:
            translator (Translator): The Translator instance.
            text (str): The text to translate.
            src (str): Source language code. Default is "en".
            dest (str): Destination language code. Default is "vi".
            max_retries (int): Maximum number of retries. Default is 3.

        Returns:
            str: Translated text or None if all retries fail.
        """
        for _ in range(max_retries):
            try:
                return translator.translate(text, src=src, dest=dest).text
            except Exception as e:
                print(f"Translation error: {e}. Retrying...")
                time.sleep(1)
        return None  # Return None if all retries fail

    def translate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a single item.

        Args:
            item (Dict[str, Any]): The item to translate.

        Returns:
            Dict[str, Any]: The translated item.
        """
        translator = Translator()
        translated_item = item.copy()

        translated_item["question_vi_ggtrans"] = self.safe_translate(
            translator, item["question"]
        )

        most_common_answer = self.get_most_common_answer(item["answers"])
        translated_item["answer_vi_ggtrans"] = self.safe_translate(
            translator, most_common_answer
        )

        translated_item["explanation_vi_ggtrans"] = [
            self.safe_translate(translator, exp) for exp in item["explanation"]
        ]

        # Remove None values from explanation list
        translated_item["explanation_vi_ggtrans"] = [
            exp for exp in translated_item["explanation_vi_ggtrans"] if exp is not None
        ]
        time.sleep(random.uniform(1, 3))
        return translated_item

    def translate_chunk(self, file_name: str, chunk: Dict[str, Dict[str, Any]], chunk_index: int) -> Dict[str, Dict[str, Any]]:
        """
        Translate a chunk of items.

        Args:
            file_name (str): The name of the file being translated.
            chunk (Dict[str, Dict[str, Any]]): The chunk of items to translate.
            chunk_index (int): The index of the chunk.

        Returns:
            Dict[str, Dict[str, Any]]: The translated chunk.
        """
        translated_chunk = {}
        for i, (key, item) in enumerate(chunk.items()):
            translated_chunk[key] = self.translate_item(item)
            sys.stdout.write(
                f"\r{file_name} Chunk {chunk_index}: {i+1}/{len(chunk)} ({(i+1)/len(chunk)*100:.2f}%)"
            )
            sys.stdout.flush()
        return translated_chunk

    def translate_batch(self, data: Dict[str, Dict[str, Any]], file_name: str, num_threads: int = 20) -> Dict[str, Dict[str, Any]]:
        """
        Translate a batch of data using multiple threads.

        Args:
            data (Dict[str, Dict[str, Any]]): The data to translate.
            file_name (str): The name of the file being translated.
            num_threads (int): The number of threads to use. Default is 20.

        Returns:
            Dict[str, Dict[str, Any]]: The translated data.
        """
        total_items = len(data)
        chunk_size = total_items // num_threads + \
            (1 if total_items % num_threads else 0)
        chunks = [
            dict(list(data.items())[i: i + chunk_size])
            for i in range(0, total_items, chunk_size)
        ]
        print(
            f"Translating {file_name} with GGTrans using {num_threads} threads...")
        translated_data = {}
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_chunk = {
                executor.submit(self.translate_chunk, file_name, chunk, i): i
                for i, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    translated_data.update(result)
                except Exception as e:
                    print(f"\nError processing chunk {chunk_index}: {str(e)}")

        print("\nggtrans translation completed.")
        return translated_data
