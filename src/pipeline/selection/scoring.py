import os
import json
import time
import openai
import argparse
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_most_common_answer(answers):
    return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]


def generate_user_content(item, sources):
    question_translations = "\n".join(
        [
            f"Translation {idx + 1}: {item[f'question_vi_{sources[idx]}']}"
            for idx in range(len(sources))
        ]
    )
    answer_translations = "\n".join(
        [
            f"Translation {idx + 1}: {item[f'answer_vi_{sources[idx]}']}"
            for idx in range(len(sources))
        ]
    )

    explanation_translations = []
    for exp_idx, exp in enumerate(item["explanation"]):
        exp_translations = "\n".join(
            [
                f"Translation {trans_idx + 1}: {item[f'explanation_vi_{sources[trans_idx]}'][exp_idx]}"
                for trans_idx in range(len(sources))
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


def prepare_batch_input_files(input_file, dataset_name, sources, batch_size=1500):
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
                user_content = generate_user_content(value, sources)
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ]

                request = {
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4",
                        "messages": messages,
                        "max_tokens": 200,
                        "temperature": 0.1,
                    },
                }

                out_f.write(json.dumps(request) + "\n")


def upload_file(file_path):
    try:
        with open(file_path, "rb") as f:
            batch_input_file = openai.files.create(file=f, purpose="fine-tune")
        return batch_input_file.id
    except Exception as e:
        print(f"Error uploading file: {e}")
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
        print(f"Error creating batch: {e}")
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
            print("No output file found.")
            return None

        file_response = openai.files.content(output_file_id)

        output_file_path = f"{batch_id}_output.jsonl"
        with open(
            os.path.join(batch_dir, output_file_path), "w", encoding="utf-8"
        ) as output_file:
            output_file.write(file_response.text)

        print(f"Results saved to {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error downloading batch results: {e}")
        return None


def parse_batch_results(result_file, original_data, sources):
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
                "question": [item[f"question_vi_{s}"] for s in sources],
                "question_scores": scores["question_scores"],
                "answer": [item[f"answer_vi_{s}"] for s in sources],
                "answer_scores": scores["answer_scores"],
                "explanation": [
                    [item[f"explanation_vi_{s}"][j] for s in sources]
                    for j in range(len(item["explanation"]))
                ],
                "explanation_scores": scores["explanation_scores"],
            }
            parsed_results.append(parsed_result)

    return parsed_results


def process_dataset(dataset_name, sources):
    dataset_file = f"vqaX_{dataset_name}_translated.json"
    dataset_path = os.path.join(datasets_dir, dataset_file)
    prepare_batch_input_files(dataset_path, dataset_name, sources)

    all_results = []

    for batch_file in os.listdir(batch_dir):
        if batch_file.startswith(f"vqaX_{dataset_name}_batch") and batch_file.endswith(
            ".jsonl"
        ):
            batch_input_file_id = upload_file(os.path.join(batch_dir, batch_file))
            if not batch_input_file_id:
                print("Error uploading batch input file.")
                continue

            batch_id = create_batch(batch_input_file_id)
            if not batch_id:
                print("Error creating batch.")
                continue

            final_status = check_batch_status(batch_id)
            if final_status != "completed":
                print(f"Batch processing failed with status: {final_status}")
                continue

            result_file = download_batch_results(batch_id)
            if not result_file:
                print("Error downloading batch results.")
                continue

            with open(dataset_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)

            batch_results = parse_batch_results(result_file, original_data, sources)
            all_results.extend(batch_results)

    final_output_file = os.path.join(
        evaluation_dir, f"{dataset_name}_gpt_evaluation.json"
    )
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Final results saved to {final_output_file}")


def check_scores_length(data):
    errors = []
    for entry in data:
        if len(entry["question_scores"]) != len(entry["question"]):
            errors.append(f"Mismatch in question_scores: {entry['question_id']}")
        if len(entry["answer_scores"]) != len(entry["answer"]):
            errors.append(f"Mismatch in answer_scores: {entry['question_id']}")
        for i, expl_scores in enumerate(entry["explanation_scores"]):
            try:
                if len(expl_scores) != len(entry["explanation"][i]):
                    errors.append(
                        f"Mismatch in explanation_scores[{i}]: {entry['question_id']}"
                    )
            except:
                errors.append(
                    f"Error in explanation_scores[{i}]: {entry['question_id']}"
                )

    return errors


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def merge_evaluation_data(**json_files):
    merged_data = {}

    for model_name, file_path in json_files.items():
        data = load_json(file_path)

        for entry in data:
            question_id = entry["question_id"]
            if question_id not in merged_data:
                merged_data[question_id] = {
                    "question_id": question_id,
                    "question": entry["question"],
                    "question_scores": {},
                    "answer": entry["answer"],
                    "answer_scores": {},
                    "explanation": entry["explanation"],
                    "explanation_scores": {},
                }

            merged_data[question_id]["question_scores"][model_name] = entry[
                "question_scores"
            ]
            merged_data[question_id]["answer_scores"][model_name] = entry[
                "answer_scores"
            ]
            merged_data[question_id]["explanation_scores"][model_name] = entry[
                "explanation_scores"
            ]

    return list(merged_data.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate translations using multiple models."
    )
    parser.add_argument("datasets_dir", help="Path to the datasets directory.")
    parser.add_argument("evaluation_dir", help="Path to the evaluation directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama", "qwen", "phi", "gemma", "gpt"],
        help="List of models to evaluate.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["train", "val", "test"],
        help="List of evaluation files.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["ggtrans", "gemini", "vinai", "gpt"],
        help="List of translation sources.",
    )

    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    evaluation_dir = args.evaluation_dir
    batch_dir = args.batch_dir
    llm_models = args.models
    evaluation_files = args.files
    sources = args.sources

    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    for dataset in evaluation_files:
        process_dataset(dataset, sources)

    for model in llm_models:
        for eval_file in evaluation_files:
            file_path = os.path.join(
                evaluation_dir, f"{eval_file}_{model}_evaluation.json"
            )
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    errors = check_scores_length(data)
                    if errors:
                        print(f"Errors in {file_path}:")
                        for error in errors:
                            print(f"  - {error}")
                    else:
                        print(f"No errors found in {eval_file}_{model}_evaluation.json")
            else:
                print(f"File {file_path} does not exist.")

    for evaluation_file in evaluation_files:
        json_files = {
            model: os.path.join(
                evaluation_dir, f"{evaluation_file}_{model}_evaluation.json"
            )
            for model in llm_models
        }

        merged_data = merge_evaluation_data(**json_files)

        output_path = os.path.join(
            evaluation_dir, f"vqaX_{evaluation_file}_evaluation.json"
        )
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(merged_data, outfile, indent=2, ensure_ascii=False)

        print(f"Merged data for {evaluation_file} saved to {output_path}")
