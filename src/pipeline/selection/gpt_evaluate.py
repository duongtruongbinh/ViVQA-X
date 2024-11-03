import os
import json
import time
import openai
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

datasets_dir = "../../../datasets"
vqax_dir = os.path.join(datasets_dir, "VQA-X")
source = ["ggtrans", "gemini", "vinai", "gpt"]
batch_dir = os.path.join(vqax_dir, "batches")
evaluation_dir = os.path.join(vqax_dir, "evaluation")
os.makedirs(batch_dir, exist_ok=True)


def get_most_common_answer(answers):
    return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]


def generate_user_content(item):
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

    # Handling explanations and their translations
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

    English Answer: {get_most_common_answer(item['answers'])}
    {answer_translations}

    {explanations_content}
    """
    return user_content.strip()


def prepare_batch_input_files(input_file, dataset_name, batch_size=1500):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    keys = list(data.keys())
    batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]

    for batch_idx, batch_keys in enumerate(batches):
        batch_output_file = f"vqaX_{dataset_name}_batch_{batch_idx + 1}.jsonl"

        with open(
            os.path.join(batch_dir, batch_output_file), "w", encoding="utf-8"
        ) as out_f:
            for key in batch_keys:
                value = data[key]
                prompt = """You will be given an English question, answer, and explanations for context. Then, you will evaluate Vietnamese translations of the question, answer, and explanation(s). Evaluate each translation based on accuracy, fluency, and cultural appropriateness, considering the full context provided. Assign a score between 0 and 100 for each translation.
                Different translations must have different scores. Ignore case differences. 
                **Return the scores exactly in the following JSON format and no additional text or explanations:**
                {{
                    "question_scores": [score_for_translation_1, score_for_translation_2, ...],
                    "answer_scores": [score_for_translation_1, score_for_translation_2, ...],
                    "explanation_scores": [
                        [score for translations of explanation 1],
                        [score for translations of explanation 2], (if has more than 1 explanation)
                        ...
                    ] (length of explanation_scores must match the number of explanations, example: n explanations -> n lists of scores in explanation_scores)
                }}
                Now, please evaluate the following:
                """
                user_content = generate_user_content(value)
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


def upload_file(file_path):
    try:
        with open(file_path, "rb") as f:
            batch_input_file = openai.files.create(file=f, purpose="batch")
        return batch_input_file.id
    except Exception as e:
        print(f"Lỗi khi tải lên tệp: {e}")
        return None


def create_batch(input_file_id):
    try:
        batch = openai.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Evaluation job"},
        )
        return batch.id
    except Exception as e:
        print(f"Lỗi khi tạo batch: {e}")
        return None


def check_batch_status(batch_id):
    while True:
        batch_status = openai.batches.retrieve(batch_id)
        status = batch_status.status
        print(f"{batch_id}: {status}", end="\r")

        if status in ["completed", "failed", "cancelled", "expired"]:
            return status
        time.sleep(60)


def download_batch_results(batch_id):
    try:
        batch_info = openai.batches.retrieve(batch_id)
        output_file_id = batch_info.output_file_id
        if not output_file_id:
            print("Không tìm thấy tệp kết quả.")
            return None

        file_response = openai.files.content(output_file_id)

        output_file_path = f"{batch_id}_output.jsonl"
        with open(
            os.path.join(batch_dir, output_file_path), "w", encoding="utf-8"
        ) as output_file:
            output_file.write(file_response.text)

        print(f"Kết quả đã được lưu vào {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Lỗi khi tải về kết quả batch: {e}")
        return None


def parse_batch_results(result_file, original_data):
    parsed_results = []

    with open(os.path.join(batch_dir, result_file), "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    for result in results:
        key = result["custom_id"]
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        scores = json.loads(content)

        if key in original_data:
            item = original_data[key]

            parsed_result = {
                "question_id": key,
                "question": [item[f"question_vi_{s}"] for s in source],
                "question_scores": scores["question_scores"],
                "answer": [item[f"answer_vi_{s}"] for s in source],
                "answer_scores": scores["answer_scores"],
                "explanation": [
                    [item[f"explanation_vi_{s}"][j] for s in source]
                    for j in range(len(item["explanation"]))
                ],
                "explanation_scores": scores["explanation_scores"],
            }
            parsed_results.append(parsed_result)

    return parsed_results


def process_dataset(dataset_name):
    dataset_file = f"vqaX_{dataset_name}_translated.json"
    dataset_path = os.path.join(vqax_dir, dataset_file)
    prepare_batch_input_files(dataset_path, dataset_name)

    all_results = []

    for batch_file in os.listdir(batch_dir):
        if batch_file.startswith(f"vqaX_{dataset_name}_batch") and batch_file.endswith(
            ".jsonl"
        ):
            batch_input_file_id = upload_file(os.path.join(batch_dir, batch_file))
            if not batch_input_file_id:
                print("Lỗi khi tải lên tệp đầu vào batch.")
                continue

            batch_id = create_batch(batch_input_file_id)
            if not batch_id:
                print("Lỗi khi tạo batch.")
                continue

            # Kiểm tra trạng thái batch
            final_status = check_batch_status(batch_id)
            if final_status != "completed":
                print(f"Batch xử lý thất bại với trạng thái: {final_status}")
                continue

            # Tải về kết quả batch
            result_file = download_batch_results(batch_id)
            if not result_file:
                print("Lỗi khi tải về kết quả batch.")
                continue

            # Đọc dữ liệu gốc
            with open(dataset_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)

            batch_results = parse_batch_results(result_file, original_data)
            all_results.extend(batch_results)

    # Lưu kết quả cuối cùng cho toàn bộ dataset
    final_output_file = os.path.join(
        evaluation_dir, f"{dataset_name}_gpt_evaluation.json"
    )
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Kết quả cuối cùng đã được lưu vào {final_output_file}")


if __name__ == "__main__":
    for dataset in ["train", "val", "test"]:
        process_dataset(dataset)
