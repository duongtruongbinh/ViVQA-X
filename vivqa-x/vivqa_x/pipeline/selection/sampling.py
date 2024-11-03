import numpy as np
from collections import Counter
from vivqa_x.pipeline.selection.utils import voting, get_softmax, divide_list_by_float, normalize_list, get_temperature_scores
np.random.seed(41)


avr_score = {
    "gpt": 0.6297,
    "llama": 0.4767,
    "gemma": 0.5787,
    "qwen": 0.5234,
    "phi": 0.4356
}

avr_score_wo_avr = {
    "gpt": 0,
    "llama": 0,
    "gemma": 0,
    "qwen": 0,
    "phi": 0
}


def sampling_method(sample_input, avr_score):
    # # minmax scaling
    # avr_score = min_max_scaling(avr_score)
    
    length = len(sample_input["question"])
    length_ex = len(sample_input["explanation"][0])  # Size of explanations

    data_point_return = {
        "question_id": sample_input["question_id"],
        "question": "None",
        "question_selection": "None",
        "answer": "None",
        "answer_selection": "None",
        "explanation": ["None"] * len(sample_input["explanation"][0]),
        "explanation_selection": "None"# Ensure size matches
    }

    temperature_score = get_temperature_score(avr_score)
    bgk_s = ["llama", "gemma", "qwen", "phi", "gpt"]
    store_choose_question = [0] * length
    store_choose_answer = [0] * length
    store_choose_explain = [[0] * length_ex for _ in range(len(sample_input["explanation"]))]  # Adjust size
    
    store_score_question = [0] * length
    store_score_answer = [0] * length
    store_score_explain = [[0] * length_ex for _ in range(len(sample_input["explanation"]))]
    
    for bgk in bgk_s:
        # question
        sample_input["question_scores"][bgk] = [float(i) for i in sample_input["question_scores"][bgk]]
        softmax_question = divide_list_by_float(normalize_list(sample_input["question_scores"][bgk]), temperature_score[bgk])
        normalized_softmax = get_softmax(softmax_question)
        # print(normalized_softmax)
        choose_question_arr = []
        for i in range(10):
            choose_question = np.random.choice(a=range(length), p=normalized_softmax)
            choose_question_arr.append(choose_question)
        count_question = Counter(choose_question_arr)
        most_common_question = count_question.most_common(1)[0][0]
        store_choose_question[most_common_question] += 1
        store_score_question[most_common_question] += avr_score[bgk]
        
        # answer
        sample_input["answer_scores"][bgk] = [float(i) for i in sample_input["answer_scores"][bgk]]
        softmax_answer = divide_list_by_float(normalize_list(sample_input["answer_scores"][bgk]), temperature_score[bgk])
        normalized_softmax = get_softmax(softmax_answer)
        choose_answer_arr = []
        for i in range(10):
            choose_answer = np.random.choice(a=range(length), p=normalized_softmax)
            choose_answer_arr.append(choose_answer)
        count_answer = Counter(choose_answer_arr)
        most_common_answer = count_answer.most_common(1)[0][0]
        store_choose_answer[most_common_answer] += 1
        store_score_answer[most_common_answer] += avr_score[bgk]
    
    for k in range(len(sample_input["explanation"])):
        for bgk in bgk_s:
            # explain
            sample_input["explanation_scores"][bgk][k] = [float(i) for i in sample_input["explanation_scores"][bgk][k]]
            softmax_explain = divide_list_by_float(normalize_list(sample_input["explanation_scores"][bgk][k]), temperature_score[bgk])
            normalized_softmax = get_softmax(softmax_explain)
            choose_explain_arr = []
            for i in range(10):
                choose_explain = np.random.choice(a=range(len(sample_input["explanation"][0])), p=normalized_softmax)
                choose_explain_arr.append(choose_explain)
            count_explain = Counter(choose_explain_arr)
            most_common_explain = count_explain.most_common(1)[0][0]
            store_choose_explain[k][most_common_explain] += 1
            store_score_explain[k][most_common_explain] += avr_score[bgk]

    # print("--------------------------------")
    # print(f"Voting final")
    # print("Voting question: ", store_choose_question)
    # print("Voting answer: ", store_choose_answer)
    # print("Voting explain: ", store_choose_explain)
    # print("score question: ", store_score_question)
    # print("score answer: ", store_score_answer)
    # print("score explain: ", store_score_explain)
    result_vote, str_return = voting(store_choose_question, store_choose_answer, store_choose_explain, store_score_question, store_score_answer, store_score_explain)
    # print("Result vote: ", result_vote)
    
    if result_vote[0] != "None":
        data_point_return["question"] = sample_input["question"][result_vote[0]]
    if result_vote[1] != "None":
        data_point_return["answer"] = sample_input["answer"][result_vote[1]]
    if result_vote[2] != "None":
        # Construct the explanation result based on the most voted index for each position
        data_point_return["explanation"] = [sample_input["explanation"][i][result_vote[2][i]] for i in range(len(result_vote[2]))]
    
    split_string = str_return.split('####')
    
    data_point_return["question_selection"] = split_string[0]
    data_point_return["answer_selection"] = split_string[1]
    data_point_return["explanation_selection"] = split_string[2]
    return data_point_return