import google.generativeai as genai
import os
import random
import time
from typing import List, Dict, Any
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from tqdm import tqdm
from .base_translator import BaseTranslator
load_dotenv()


class GeminiTranslator(BaseTranslator):
    def __init__(self):
        """
        Initialize the GeminiTranslator with required dependencies.
        """
        super().__init__()
        self.api_keys = os.getenv("GEMINI_APIKEYS").split(",")
        self.current_key_index = 0
        self.requirements = []
        self.batch_size = 4

    def get_batch_prompt(self, sample_list: List[Dict[str, Any]]) -> str:
        """
        Generate a batch prompt for translation.

        Args:
            sample_list (List[Dict[str, Any]]): List of samples to translate.

        Returns:
            str: The generated batch prompt.
        """
        prompt_intro = (
            "You are a Vietnamese translator. Translate the following input strings into Vietnamese. "
            "The inputs include questions, answers, and explanations. Provide the translation directly without any additional commentary or analysis.\n"
            "Note: Return only the string of the translation for each input, separated by newlines. For example:\n"
            "Example:\n"
            "Input 1: Question: Is the window open?\nAnswer: no\nExplanation: the window shutters are closed\nExplanation: It's nighttime\n"
            "Input 2: Question: What color is the sky?\nAnswer: blue\nExplanation: it's a clear day\n"
            "Input 3: Question: How many dogs are there?\nAnswer: two\nExplanation: I can see two dogs in the park\nExplanation: They are playing fetch\n"
            "Output:\n"
            "Cửa sổ có mở không?\n"
            "không\n"
            "các cánh cửa sổ đóng\n"
            "Trời đã tối\n"
            "Bầu trời màu gì?\n"
            "xanh\n"
            "đó là một ngày quang đãng\n"
            "Có bao nhiêu con chó?\n"
            "hai\n"
            "Tôi có thể thấy hai con chó trong công viên\n"
            "Chúng đang chơi ném bắt\n"
            "These are the inputs:\n"
        )

        prompt_inputs = ""
        for i, sample in enumerate(sample_list, 1):
            prompt_inputs += (
                f"Input {i}: Question: {sample['question']}\nAnswer: {sample['answer']}\n"
            )
            for exp in sample["explanation"]:
                prompt_inputs += f"Explanation: {exp}\n"
            prompt_inputs += "\n"

        return prompt_intro + prompt_inputs + "Output:\n"

    def translate_single_batch(self, samples: List[Dict[str, Any]], api_key: str) -> List[str]:
        """
        Translate a single batch of samples.

        Args:
            samples (List[Dict[str, Any]]): List of samples to translate.
            api_key (str): The API key to use for translation.

        Returns:
            List[str]: List of translated strings.
        """
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_batch = self.get_batch_prompt(samples)
        try:
            response = model.generate_content(
                prompt_batch,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            translations = [t for t in response.text.strip().split("\n") if t]
            expected_translations = sum(
                len(sample["explanation"]) for sample in samples
            ) + 2 * len(samples)
            if len(translations) == expected_translations:
                return translations
            else:
                print(
                    f"Error: Expected {expected_translations} translations but got {len(translations)} translations."
                )
            return None
        except Exception as e:
            print(f"Error during translation: {e}")
            return None

    def translate_batch(self, data: Dict[str, Dict[str, Any]], file_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Translate a batch of data.

        Args:
            data (Dict[str, Dict[str, Any]]): The data to translate.
            file_name (str): The name of the file being translated.

        Returns:
            Dict[str, Dict[str, Any]]: The translated data.
        """
        processed_data = {}
        batch_size = self.batch_size
        keys = list(data.keys())
        for i in tqdm(
            range(0, len(keys), batch_size),
            desc=f"Translating {file_name} with Gemini",
        ):
            batch_keys = keys[i: i + batch_size]
            batch_samples = [
                {
                    "question": data[key]["question"],
                    "answer": self.get_most_common_answer(data[key]["answers"]),
                    "explanation": data[key]["explanation"],
                }
                for key in batch_keys
            ]
            api_key = self.api_keys[self.current_key_index %
                                    len(self.api_keys)]
            translations = self.translate_single_batch(batch_samples, api_key)
            self.current_key_index = (
                self.current_key_index + 1) % len(self.api_keys)
            if translations:
                translation_index = 0
                for key, sample in zip(batch_keys, batch_samples):
                    processed_data[key] = data[key].copy()
                    ques_idx, ans_idx, expl_idx = (
                        translation_index,
                        translation_index + 1,
                        translation_index + 2,
                    )
                    processed_data[key]["question_vi"] = translations[ques_idx].strip(
                    )
                    processed_data[key]["answer_vi"] = translations[ans_idx].strip()
                    processed_data[key]["explanation_vi"] = translations[
                        expl_idx: expl_idx + len(sample["explanation"])
                    ]
                    translation_index += 2 + len(sample["explanation"])
            time.sleep(random.uniform(5, 10))  # Pause between requests
            if i % (100 * batch_size) == 0 and i > 0:
                # Longer pause between larger chunks
                time.sleep(random.randint(50, 200))
        return processed_data
