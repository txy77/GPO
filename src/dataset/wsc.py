import os
import json


GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)



def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            json_object = json.loads(line)
            data.append(json_object)
    return data

class WSC_Dataset:

    def get_ratio(self):
        train_ratio = 1  
        eval_ratio = 1  
        test_ratio = 1  
        print(
            f"[Train ratio]: {train_ratio}, [Eval ratio]: {eval_ratio}, [Test ratio]: {test_ratio}"
        )
        return train_ratio, eval_ratio, test_ratio
    
    def read_data(self, task):

        data = []

        data_folder_path = os.path.join(
            GPO_ROOT_PATH, f"data/common-nlp/WSC"
        )

        f_train = os.path.join(data_folder_path, f"train.jsonl")
        f_eval = os.path.join(data_folder_path, f"eval.jsonl")
        f_test = os.path.join(data_folder_path, f"test.jsonl")
        f_few_shot_examples = os.path.join(data_folder_path, f"few_shot_examples.jsonl")
        f_format = os.path.join(data_folder_path, f"format.jsonl")

        train_data = read_jsonl(f_train)
        eval_data = read_jsonl(f_eval)
        test_data = read_jsonl(f_test)
        few_shot_examples = read_jsonl(f_few_shot_examples)
        format_data = read_jsonl(f_format)

        train_num_examples = len(train_data)
        eval_num_examples = len(eval_data)
        test_num_examples = len(test_data)
        few_shot_num_examples = len(few_shot_examples)
        format_num_examples = len(format_data)


        data.append({
            "task": task,
            "train_data": train_data,
            "eval_data": eval_data,
            "test_data": test_data,
            "few_shot_data": few_shot_examples,
            "format_data": format_data,
            "train_num_examples": train_num_examples,
            "eval_num_examples": eval_num_examples,
            "test_num_examples": test_num_examples,
            "few_shot_num_examples": few_shot_num_examples,
            "format_num_examples": format_num_examples,
        })

        return data
    
    def get_single_question(self, data, idx):
        return data[idx]["input"]
    
    def get_single_answer(self, data, idx):
        return data[idx]["output"]
    
    def get_single_solution(self, data, idx):
        pass
    

    def get_task_setting(self, task):
        return {
            "is_multiple_choice": True,
            "prediction_treat_as_number": False,
            "prediction_treat_as_bool": False,
            "prediction_treat_as_rouge": False,
            "extract_final_answer_by_prompting_again": True,
        }