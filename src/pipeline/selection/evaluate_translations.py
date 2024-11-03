import os
import json
import torch
from transformers import pipeline
import regex as re
import argparse
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from collections import Counter

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["WORLD_SIZE"] = "2"
torch.random.manual_seed(0)

DATASETS_DIR = "../../../datasets"
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

generation_args = {
    "max_new_tokens": 500,
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.9,
    "num_return_sequences": 1,
    "pad_token_id": 50256,
}


def create_pipeline(model_id, **kwargs):
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        token=HF_ACCESS_TOKEN,
        device_map="auto",
        **kwargs,
    )


def load_model(model_name):
    model_configs = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gemma": "google/gemma-2-9b-it",
        "phi": "microsoft/Phi-3-medium-4k-instruct",
        "qwen": "Qwen/Qwen2-7B-Instruct",
    }

    if model_name in model_configs:
        return create_pipeline(model_configs[model_name])
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_evaluation_prompt(
    question: str,
    answer: str,
    explanation: str,
    eval_type: str,
    translations: list[str],
) -> str:
    prompt = (
        f"""
You will evaluate {len(translations)} Vietnamese translations of the given {eval_type} from a Visual Question Answering (VQA) task. The context includes an English question, answer, and explanation. Your evaluation should be based on three criteria: accuracy, fluency, and cultural fit. 

Important:
- Do not judge the correctness of the answer itself, as it is based on the image and not the question.
- Ignore case differences when evaluating translations.
- Identical translations should receive the same score.
- Provide a score from 0 to 100 for each translation, one score per line, without any explanations.

Example:
Question: What is the capital of France?
Answer: The capital of France is Paris.
Explanation: Paris has been the capital of France since the Middle Ages.

Translations:
1: Thủ đô của nước Pháp là gì?
2: Thủ đô của Pháp là gì?

Output:
95
90

Now evaluate the following:

Question: {question}
Answer: {answer}
Explanation: {explanation}

Translations:
"""
        + "\n".join(f"{i+1}: {t}" for i, t in enumerate(translations))
        + """

Output:
"""
    )
    return prompt


def evaluate_translations(
    model_name,
    model,
    question: str,
    answer: str,
    explanation: str,
    eval_type: str,
    translations: list[str],
) -> list[int]:
    prompt = create_evaluation_prompt(
        question, answer, explanation, eval_type, translations
    )
    try:
        with torch.no_grad():
            if model_name in ["llama", "phi", "qwen"]:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that can complete a given instruction.",
                    },
                    {"role": "user", "content": prompt},
                ]
                response = model(messages, **generation_args)
                generated_text = response[0]["generated_text"]
                generated_text = generated_text[-1]["content"]
            elif model_name == "gemma":
                messages = [
                    {
                        "role": "user",
                        "content": "You are a helpful AI assistant that can complete a given instruction."
                        + prompt,
                    }
                ]
                response = model(messages, **generation_args)
                generated_text = response[0]["generated_text"]
                generated_text = generated_text[-1]["content"]

            scores = re.findall(r"\d+", generated_text)
            scores = [int(score.strip()) for score in scores]
            return scores
    except Exception as e:
        print(f"Error evaluating translations: {e}")
        return []


def get_most_common_answer(answers):
    return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]


def process_dataset(model_name, model, data, output_file):
    results = []
    sources = ["ggtrans", "gemini", "vinai", "gpt"]
    for sample_id, sample in tqdm(data.items(), desc=f"Processing {model_name}"):
        common_answer = get_most_common_answer(sample["answers"])
        question_translations = [sample[f"question_vi_{source}"] for source in sources]
        answer_translations = [sample[f"answer_vi_{source}"] for source in sources]
        explanation_translations = [
            [sample[f"explanation_vi_{source}"][i] for source in sources]
            for i in range(len(sample["explanation"]))
        ]

        question_scores = evaluate_translations(
            model_name,
            model,
            sample["question"],
            common_answer,
            sample["explanation"][0],
            "question",
            question_translations,
        )
        answer_scores = evaluate_translations(
            model_name,
            model,
            sample["question"],
            common_answer,
            sample["explanation"][0],
            "answer",
            answer_translations,
        )
        explanation_scores = [
            evaluate_translations(
                model_name,
                model,
                sample["question"],
                common_answer,
                sample["explanation"][i],
                "explanation",
                exp_translations,
            )
            for i, exp_translations in enumerate(explanation_translations)
        ]

        result = {
            "question_id": sample_id,
            "question": question_translations,
            "question_scores": question_scores,
            "answer": answer_translations,
            "answer_scores": answer_scores,
            "explanation": explanation_translations,
            "explanation_scores": explanation_scores,
        }

        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main(model_name):
    datasets_dir = "../../../datasets"
    vqax_dir = os.path.join(datasets_dir, "VQA-X")
    train_path = os.path.join(vqax_dir, "vqaX_train_translated.json")
    test_path = os.path.join(vqax_dir, "vqaX_test_translated.json")
    val_path = os.path.join(vqax_dir, "vqaX_val_translated.json")

    output_dir = os.path.join(vqax_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_name)
    print(f"Model {model_name} loaded successfully.")

    dataset_files = {
        "val": val_path,
        "train": train_path,
        "test": test_path,
    }
    for dataset_name, file_path in dataset_files.items():
        with open(file_path) as f:
            data = json.load(f)

        output_file = os.path.join(
            output_dir, f"{dataset_name}_{model_name}_evaluation.json"
        )
        process_dataset(model_name, model, data, output_file)
        print(
            f"Finished processing {dataset_name} dataset. Results saved to {output_file}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate translations using a specified model."
    )
    parser.add_argument(
        "model",
        choices=["llama", "gemma", "phi", "qwen"],
        help="The model to use for evaluation.",
    )
    args = parser.parse_args()

    main(args.model)
