from vivqa_x.pipeline.selection.utils import voting
import numpy as np
np.random.seed(41)

def argmax_method(sample_input, avr_score):
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
    
    bgk_s = ["llama", "gemma", "qwen", "phi", "gpt"]
    store_choose_question = [0] * length
    store_choose_answer = [0] * length
    store_choose_explain = [[0] * length_ex for _ in range(len(sample_input["explanation"]))]  # Adjust size
    
    store_score_question = [0] * length
    store_score_answer = [0] * length
    store_score_explain = [[0] * length_ex for _ in range(len(sample_input["explanation"]))]    

    for bgk in bgk_s:
        # question
        choose_question = sample_input["question_scores"][bgk].index(max(sample_input["question_scores"][bgk]))
        store_choose_question[choose_question] += 1
        store_score_question[choose_question] += avr_score[bgk]
        
        # answer
        choose_answer = sample_input["answer_scores"][bgk].index(max(sample_input["answer_scores"][bgk]))
        store_choose_answer[choose_answer] += 1
        store_score_answer[choose_answer] += avr_score[bgk]

    for k in range(len(sample_input["explanation"])):
        for bgk in bgk_s:
            # explain
            choose_explain = sample_input["explanation_scores"][bgk][k].index(max(sample_input["explanation_scores"][bgk][k]))
            store_choose_explain[k][choose_explain] += 1
            store_score_explain[k][choose_explain] += avr_score[bgk]

    
    # print("--------------------------------")
    # print(f"Voting final")
    # print("Voting question: ", store_choose_question)
    # print("Voting answer: ", store_choose_answer)
    # print("Voting explain: ", store_choose_explain)

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