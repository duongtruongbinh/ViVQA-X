from googletrans import Translator
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time
from collections import Counter
import random


def get_most_common_answer(answers):
    return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]


def safe_translate(translator, text, src="en", dest="vi", max_retries=3):
    for _ in range(max_retries):
        try:
            return translator.translate(text, src=src, dest=dest).text
        except Exception as e:
            print(f"Translation error: {e}. Retrying...")
            time.sleep(1)
    return None  # Return None if all retries fail


def translate_item(item):
    translator = Translator()
    translated_item = item.copy()

    translated_item["question_vi_ggtrans"] = safe_translate(
        translator, item["question"]
    )

    most_common_answer = get_most_common_answer(item["answers"])
    translated_item["answer_vi_ggtrans"] = safe_translate(
        translator, most_common_answer
    )

    translated_item["explanation_vi_ggtrans"] = [
        safe_translate(translator, exp) for exp in item["explanation"]
    ]

    # Remove None values from explanation list
    translated_item["explanation_vi_ggtrans"] = [
        exp for exp in translated_item["explanation_vi_ggtrans"] if exp is not None
    ]
    time.sleep(random.uniform(1, 3))
    return translated_item


def translate_chunk(chunk, chunk_index):
    translated_chunk = {}
    for i, (id, item) in enumerate(chunk.items()):
        try:
            translated_item = translate_item(item)
            if all(
                v is not None
                for v in [
                    translated_item["question_vi_ggtrans"],
                    translated_item["answer_vi_ggtrans"],
                ]
                + translated_item["explanation_vi_ggtrans"]
            ):
                translated_chunk[id] = translated_item
            else:
                print(
                    f"\nWarning: Item {id} in chunk {chunk_index} was not fully translated."
                )
            sys.stdout.write(
                f"\rChunk {chunk_index}: {i+1}/{len(chunk)} ({(i+1)/len(chunk)*100:.2f}%)"
            )
            sys.stdout.flush()
        except Exception as e:
            print(f"\nError translating item {id} in chunk {chunk_index}: {str(e)}")
    return translated_chunk


def translate_batch(data, file_type: str, num_threads=20):
    total_items = len(data)
    chunk_size = total_items // num_threads + (1 if total_items % num_threads else 0)
    chunks = [
        dict(list(data.items())[i : i + chunk_size])
        for i in range(0, total_items, chunk_size)
    ]

    translated_data = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_chunk = {
            executor.submit(translate_chunk, chunk, i): i
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
