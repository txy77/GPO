import os
import json


GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

mmlu_tasks = [
    'abstract_algebra', 
    'anatomy', 
    'astronomy', 
    'business_ethics', 
    'clinical_knowledge', 
    'college_biology', 
    'college_chemistry', 
    'college_computer_science', 
    'college_mathematics', 
    'college_medicine', 
    'college_physics', 
    'computer_security', 
    'conceptual_physics', 
    'econometrics', 
    'electrical_engineering', 
    'elementary_mathematics', 
    'formal_logic', 
    'global_facts', 
    'high_school_biology', 
    'high_school_chemistry', 
    'high_school_computer_science', 
    'high_school_european_history', 
    'high_school_geography', 
    'high_school_government_and_politics', 
    'high_school_macroeconomics', 
    'high_school_mathematics', 
    'high_school_microeconomics', 
    'high_school_physics', 
    'high_school_psychology', 
    'high_school_statistics', 
    'high_school_us_history', 
    'high_school_world_history', 
    'human_aging', 
    'human_sexuality', 
    'international_law', 
    'jurisprudence', 
    'logical_fallacies', 
    'machine_learning', 
    'management', 
    'marketing', 
    'medical_genetics', 
    'miscellaneous', 
    'moral_disputes', 
    'moral_scenarios', 
    'nutrition', 
    'philosophy', 
    'prehistory', 
    'professional_accounting', 
    'professional_law', 
    'professional_medicine', 
    'professional_psychology', 
    'public_relations', 
    'security_studies', 
    'sociology', 
    'us_foreign_policy', 
    'virology', 
    'world_religions'
    ]




def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            json_object = json.loads(line)
            data.append(json_object)
    return data

class MMLU_Dataset:

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
            GPO_ROOT_PATH, f"data/MMLU/"
        )
        
        if task[0] == "all":
            tasks = mmlu_tasks
        elif isinstance(task, str):
            assert task in mmlu_tasks
            tasks = [task]
        elif isinstance(task, list):
            for t in task:
                assert t in mmlu_tasks
            tasks = task
            
        for task in tasks:
            task_data_folder_path = os.path.join(
                GPO_ROOT_PATH, f"data/MMLU/{task}"
            )
            
            f_train = os.path.join(task_data_folder_path, f"train.jsonl")
            f_eval = os.path.join(task_data_folder_path, f"eval.jsonl")
            f_test = os.path.join(task_data_folder_path, f"test.jsonl")
            f_few_shot_examples = os.path.join(task_data_folder_path, f"few_shot_examples.jsonl")
            f_format = os.path.join(task_data_folder_path, f"format.jsonl")
            
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

            datas.append({
                # task
                "task": task,
                # data
                "train_data": train_data,
                "eval_data": eval_data,
                "test_data": test_data,
                "few_shot_data": few_shot_examples,
                "format_data": format_data,
                # the number of data
                "train_num_examples": train_num_examples,
                "eval_num_examples": eval_num_examples,
                "test_num_examples": test_num_examples,
                "few_shot_num_examples": few_shot_num_examples,
                "format_num_examples": format_num_examples,
            })

        return datas
    
    def get_single_question(self, data, idx):
        return data[idx]["input"]
    
    def get_single_answer(self, data, idx):
        return data[idx]["output"]
    
    def get_single_solution(self, data, idx):
        pass
    
    
    def get_q(self, data):
        return data["input"]
    
    def get_a(self, data):
        return data["output"]

    def get_task_setting(self, task):
        return {
            "is_multiple_choice": True,
            "prediction_treat_as_number": False,
            "prediction_treat_as_bool": False,
            "prediction_treat_as_rouge": False,
            "extract_final_answer_by_prompting_again": True,
        }