import csv
import numpy as np


def store_csv(data, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["question_id", "question", "question_selection", "answer", "answer_selection", "explanation", "explanation_selection"])
        writer.writeheader()

        # Convert the 'explain' list to a string and write to the CSV file
        for entry in data:
            entry['explanation'] = '###'.join(entry['explanation'])
            writer.writerow(entry)


def voting(store_choose_question, store_choose_answer, store_choose_explain, store_score_question, store_score_answer, store_score_explain):
    indices = []
    str_return = ""
    # print("voting")
    # print(store_choose_question, store_choose_answer)
    # Handle questions and answers
    
    max_value = max(store_choose_question)
    if store_choose_question.count(max_value) == 1:
        indices.append(store_choose_question.index(max_value))
        str_return = str_return + process_index(store_choose_question.index(max_value)) + "####"
    else:
        index = get_max_indices_in_both(store_choose_question, store_score_question)
        indices.append(index)
        str_return = str_return + process_index(index) + "####"
    
    max_value = max(store_choose_answer)
    if store_choose_answer.count(max_value) == 1:
        indices.append(store_choose_answer.index(max_value))
        str_return = str_return + process_index(store_choose_answer.index(max_value)) + "####"
    else:
        # In case of a tie, select the index of the first occurrence
        index = get_max_indices_in_both(store_choose_answer, store_score_answer)
        indices.append(index)
        str_return = str_return + process_index(index) + "####"    
    
    explain_indices = []
    sub_return = ""
    for votes, scores in zip(store_choose_explain, store_score_explain):
        max_value = max(votes)
        if votes.count(max_value) == 1:
            explain_indices.append(votes.index(max_value))
            sub_return = sub_return + process_index(votes.index(max_value)) + "###"
        else:
            # In case of a tie, select the index of the first occurrence
            index = get_max_indices_in_both(votes, scores)
            explain_indices.append(index)
            sub_return = sub_return + process_index(index) + "###"  
    sub_return = sub_return[:-3]
    str_return = str_return + sub_return
    print(str_return)
    indices.append(explain_indices)
    return indices, str_return


def get_max_indices_in_both(arr1, arr2):
    max_value_in_arr1 = max(arr1)
    indices_max_arr1 = [i for i, val in enumerate(arr1) if val == max_value_in_arr1]
    max_index_in_arr2 = max(indices_max_arr1, key=lambda i: arr2[i])
    return max_index_in_arr2

def process_index(index):
    if index == 0:
        return "ggtrans"
    if index == 1:
        return "gemini"
    if index == 2:
        return "vinai"
    if index == 3:
        return "gpt"
    
def normalize_list(input_list):
    # Find the minimum and maximum values in the list
    min_val = 0
    max_val = 100
    
    # Calculate the range
    range_val = max_val - min_val
    
    # Normalize the list
    normalized_list = [(x - min_val) / (range_val) for x in input_list]
    
    return normalized_list

def get_softmax(arr):
    arr = np.array(arr)
    exp_arr = np.exp(arr - np.max(arr))  # For numerical stability
    return exp_arr / np.sum(exp_arr)

def divide_list_by_float(input_list, divisor):
    # Divide each element of the list by the divisor
    divided_list = [x / (divisor) for x in input_list]
    return divided_list

def get_temperature_score(avr_score):
    return {key: round(1 - value, 2) for key, value in avr_score.items()}