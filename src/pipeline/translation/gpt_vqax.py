# gpt_vqax.py
import os
import json
import time
import openai
from dotenv import load_dotenv
from collections import Counter
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def prepare_batch_file(data, batch_file):
    """Prepare batch file for OpenAI processing"""
    with open(batch_file, "w") as out_f:
        for key, value in data.items():
            common_answer = Counter(
                ans["answer"] for ans in value["answers"]
            ).most_common(1)[0][0]
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


def process_batch(batch_file):
    """Process batch through OpenAI API"""
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


def download_results(batch_id):
    """Download and parse batch results"""
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


def translate_batch(data, file_type):
    """Main translation function that processes the entire dataset"""
    translated_dataset = {}

    # Create temporary batch file
    batch_file = f"vqaX_{file_type}_gpt_batch.jsonl"
    prepare_batch_file(data, batch_file)

    # Process batch
    batch_id, status = process_batch(batch_file)
    if not batch_id:
        print(f"Batch processing failed with status: {status}")
        return translated_dataset

    # Download and process results
    results = download_results(batch_id)
    if not results:
        print("Failed to download results")
        return translated_dataset

    # Parse results
    for result in tqdm(results, desc=f"Processing {file_type} dataset with GPT"):
        try:
            key = result["custom_id"]
            content = result["response"]["body"]["choices"][0]["message"]["content"]

            if "\n" in content:
                translations = [t.strip() for t in content.split("\n") if t.strip()]
            elif "|" in content:
                translations = [t.strip() for t in content.split("|") if t.strip()]
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
