import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
from typing import List, Dict, Any
from .base_translator import BaseTranslator


class VinaiTranslator(BaseTranslator):
    def __init__(self):
        """
        Initialize the VinaiTranslator with required dependencies.
        """
        super().__init__()
        self.requirements = []
        self.tokenizer_en2vi = AutoTokenizer.from_pretrained(
            "vinai/vinai-translate-en2vi-v2", src_lang="en_XX"
        )
        self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(
            "vinai/vinai-translate-en2vi-v2")
        self.device = torch.device("cuda")
        self.model_en2vi.to(self.device)
        self.model_en2vi.eval()
        self.set_seed()

    def translate_en2vi(self, en_texts: List[str]) -> List[str]:
        """
        Translate a list of English texts to Vietnamese.

        Args:
            en_texts (List[str]): List of English texts to translate.

        Returns:
            List[str]: List of translated Vietnamese texts.
        """
        input_ids = self.tokenizer_en2vi(
            en_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        output_ids = self.model_en2vi.generate(
            **input_ids,
            decoder_start_token_id=self.tokenizer_en2vi.convert_tokens_to_ids(
                "vi_VN"),
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True,
        )
        vi_texts = self.tokenizer_en2vi.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return vi_texts

    def split_translated_text(self, text: str) -> Dict[str, Any]:
        """
        Split translated text into question, answer, and explanations.

        Args:
            text (str): The translated text.

        Returns:
            Dict[str, Any]: A dictionary with keys 'question', 'answer', and 'explanation'.
        """
        text = re.sub(r"Câu trả lời:", "Trả lời:", text, flags=re.IGNORECASE)
        text = re.sub(r"Lời giải thích:", "Giải thích:",
                      text, flags=re.IGNORECASE)
        parts = re.split(r"(Câu hỏi:|Trả lời:|Giải thích:)",
                         text, flags=re.IGNORECASE)
        result = {"question": "", "answer": "", "explanation": []}
        current_key = ""
        for part in parts:
            part = part.strip()
            lower_part = part.lower()
            if lower_part in ["câu hỏi:", "trả lời:", "giải thích:"]:
                if lower_part == "câu hỏi:":
                    current_key = "question"
                elif lower_part == "trả lời:":
                    current_key = "answer"
                elif lower_part == "giải thích:":
                    current_key = "explanation"
            elif current_key:
                if current_key == "explanation":
                    explanations = re.split(r"(?<=[.!?])\s+", part)
                    result[current_key].extend(
                        [exp.strip() for exp in explanations if exp.strip()]
                    )
                else:
                    result[current_key] = part.strip()
        # Split answer and explanation if they are combined
        if not result["explanation"] and ". " in result["answer"]:
            answer_parts = result["answer"].split(". ", 1)
            if len(answer_parts) == 2:
                result["answer"] = answer_parts[0].strip()
                result["explanation"] = [answer_parts[1].strip()]
        return result

    def translate_batch(self, data: Dict[str, Dict[str, Any]], file_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Translate a batch of data.

        Args:
            data (Dict[str, Dict[str, Any]]): The data to translate.
            file_name (str): The name of the file being translated.

        Returns:
            Dict[str, Dict[str, Any]]: The translated data.
        """
        translated_dataset = {}
        batch_size = 30
        dataset_items = list(data.items())
        for i in tqdm(
            range(0, len(dataset_items), batch_size),
            desc=f"Translating {file_name} with VinAI",
        ):
            batch = dataset_items[i: i + batch_size]
            # Prepare texts for translation
            qa_texts = []
            explanation_texts = []
            for key, item in batch:
                question = item["question"]
                most_common_answer = self.get_most_common_answer(
                    item["answers"])
                qa_text = f"Question: {question} Answer: {most_common_answer}"
                qa_texts.append(qa_text)
                for explanation in item["explanation"]:
                    explanation_texts.append(explanation)
            # Translate questions and answers
            translated_qa_texts = self.translate_en2vi(qa_texts)
            # Translate explanations
            translated_explanation_texts = self.translate_en2vi(
                explanation_texts)
            explanation_index = 0
            for j, (key, item) in enumerate(batch):
                translated_item = item.copy()
                split_result = self.split_translated_text(
                    translated_qa_texts[j])
                translated_item["question_vi"] = split_result["question"]
                translated_item["answer_vi"] = split_result["answer"]
                translated_item["explanation_vi"] = []
                for _ in range(len(item["explanation"])):
                    if explanation_index < len(translated_explanation_texts):
                        translated_item["explanation_vi"].append(
                            translated_explanation_texts[explanation_index]
                        )
                        explanation_index += 1
                translated_dataset[key] = translated_item
        return translated_dataset
