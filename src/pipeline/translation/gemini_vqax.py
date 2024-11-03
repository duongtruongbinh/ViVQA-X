import google.generativeai as genai
import os
import json
import random
import time
from collections import Counter
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
GEMINI_APIKEYS = os.getenv("GEMINI_APIKEYS").split(",")
current_key_index = 0


def get_most_common_answer(answers):
    return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]


def get_batch_prompt(sample_list):
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


def translate_single_batch(samples, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt_batch = get_batch_prompt(samples)
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
        print(f"Error in translate_batch: {e}")
        return None


def translate_batch(data, file_type: str):
    global current_key_index
    processed_data = {}
    batch_size = 4
    keys = list(data.keys())

    for i in tqdm(
        range(0, len(keys), batch_size),
        desc=f"Translating {file_type} dataset with Gemini",
    ):
        batch_keys = keys[i : i + batch_size]
        batch_samples = [
            {
                "question": data[key]["question"],
                "answer": get_most_common_answer(data[key]["answers"]),
                "explanation": data[key]["explanation"],
            }
            for key in batch_keys
        ]

        api_key = GEMINI_APIKEYS[current_key_index % len(GEMINI_APIKEYS)]
        translations = translate_single_batch(batch_samples, api_key)
        current_key_index = (current_key_index + 1) % len(GEMINI_APIKEYS)

        if translations:
            translation_index = 0
            for key, sample in zip(batch_keys, batch_samples):
                processed_data[key] = data[key].copy()
                ques_idx, ans_idx, expl_idx = (
                    translation_index,
                    translation_index + 1,
                    translation_index + 2,
                )
                processed_data[key]["question_vi"] = translations[ques_idx].strip()
                processed_data[key]["answer_vi"] = translations[ans_idx].strip()
                processed_data[key]["explanation_vi"] = translations[
                    expl_idx : expl_idx + len(sample["explanation"])
                ]
                translation_index += 2 + len(sample["explanation"])

        time.sleep(random.uniform(6, 12))  # Pause between requests
        if i % (100 * batch_size) == 0 and i > 0:
            time.sleep(random.randint(60, 200))  # Longer pause between larger chunks

    return processed_data
