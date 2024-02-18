import os
import json
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

bbh_tasks = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

numerical_output_tasks = {
    "object_counting",
    "multistep_arithmetic_two",
}

multiple_choice_tasks = {
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
}

boolean_tasks = {
    "boolean_expressions",  # True or False
    "causal_judgement",  # yes or no
    "formal_fallacies",  # valid or invalid
    "navigate",  # yes or no
    "sports_understanding",  # yes or no
    "web_of_lies",  # yes or no
}

def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            json_object = json.loads(line)
            data.append(json_object)

    return data


class BBH_Dataset:
    
    def get_ratio(self):
        train_ratio = 1  
        eval_ratio = 1  
        test_ratio = 1  
        print(
            f"[Train ratio]: {train_ratio}, [Eval ratio]: {eval_ratio}, [Test ratio]: {test_ratio}"
        )
        return train_ratio, eval_ratio, test_ratio

    def read_data(self, task):
        
        datas = []
        
        root_data_folder_path = os.path.join(
            GPO_ROOT_PATH, f"data/BIG-Bench-Hard/"
        )
        
        if task[0] == "all":
            tasks = bbh_tasks
        elif isinstance(task, str):
            assert task in bbh_tasks
            tasks = [task]
        elif isinstance(task, list):
            for t in task:
                assert t in bbh_tasks
            tasks = task
            
        for task in tasks:
            task_data_folder_path = os.path.join(
                GPO_ROOT_PATH, f"data/BIG-Bench-Hard/{task}"
            )
            f_bbh_task_train = os.path.join(task_data_folder_path, f"train.jsonl")
            f_bbh_task_eval = os.path.join(task_data_folder_path, f"eval.jsonl")
            f_bbh_task_test = os.path.join(task_data_folder_path, f"test.jsonl")
            f_bbh_task_format = os.path.join(task_data_folder_path, f"format.jsonl")
            f_bbh_task_few_shot_examples = os.path.join(
                task_data_folder_path, f"few_shot_examples.jsonl"
            )

            bbh_task_train_data = read_jsonl(f_bbh_task_train)
            bbh_task_eval_data = read_jsonl(f_bbh_task_eval)
            bbh_task_test_data = read_jsonl(f_bbh_task_test)
            bbh_task_format_data = read_jsonl(f_bbh_task_format)
            bbh_task_few_shot_data = read_jsonl(f_bbh_task_few_shot_examples)
            
            train_num_examples = len(bbh_task_train_data)
            eval_num_examples = len(bbh_task_eval_data)
            test_num_examples = len(bbh_task_test_data)
            format_num_examples = len(bbh_task_format_data)
            few_shot_num_examples = len(bbh_task_few_shot_data)

            # print(f"[Train] number of examples in the current task: {train_num_examples}")
            # print(f"[eval] number of examples in the current task: {eval_num_examples}")
            # print(f"[Test] number of examples in the current task: {test_num_examples}")
            # print(f"[Format] number of examples in the current task: {format_num_examples}")
            # print(
            #     f"[Few Shot Examples] number of examples in the current task: {few_shot_num_examples}"
            # )
            
            datas.append({
                # task
                "task": task,
                # data
                "train_data": bbh_task_train_data,
                "eval_data": bbh_task_eval_data,
                "test_data": bbh_task_test_data,
                "format_data": bbh_task_format_data,
                "few_shot_data": bbh_task_few_shot_data,
                # the number of data
                "train_num_examples": train_num_examples,
                "eval_num_examples": eval_num_examples,
                "test_num_examples": test_num_examples,
                "format_num_examples": format_num_examples,
                "few_shot_num_examples": few_shot_num_examples,
            })

        return datas

    def get_single_question(self, data, idx):
        question = data[idx]["input"]
        return question

    def get_single_answer(self, data, idx):
        true_answer = data[idx]["target"]
        return true_answer

    def get_single_solution(self, data, idx):
        pass
    
    def get_q(self, data):
        return data["input"]
    
    def get_a(self, data):
        return data["target"]

    def get_task_setting(self, task):
        
        is_multiple_choice = bool(task in multiple_choice_tasks)
        prediction_treat_as_number = bool(task in numerical_output_tasks)
        prediction_treat_as_bool = bool(task in boolean_tasks)
        prediction_treat_as_rouge = False
        extract_final_answer_by_prompting_again = True
        return {
            "is_multiple_choice": is_multiple_choice,
            "prediction_treat_as_number": prediction_treat_as_number,
            "prediction_treat_as_bool": prediction_treat_as_bool,
            "prediction_treat_as_rouge": prediction_treat_as_rouge,
            "extract_final_answer_by_prompting_again": extract_final_answer_by_prompting_again,
        }
