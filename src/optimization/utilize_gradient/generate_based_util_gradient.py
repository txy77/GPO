import os
import sys
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.dataset.base import Base_Dataset

class Generate_based_util_gradient:
    def __init__(self, dataset_name, data, include_qa, instruction_pos, format_index) -> None:
        self.Dataset_class = Base_Dataset(dataset_name)
        self.data = data
        self.instruction_pos = instruction_pos
        self.format_index = format_index
        self.include_qa = include_qa
        
        assert self.instruction_pos in {
            "before_Q",
            "Q_begin",
            "Q_end",
            "A_begin",
        }
        
    def gen_util_gradient_text(self, current_prompt, current_prompt_score, gradient, k_list_string=None, real_time_para_momentum=[]):
        
        # "This new prompt is concise, effective and generally applicable to all QA pairs.\n"
        
        if self.instruction_pos == 'A_begin':
            task_example_list = [
                f"Input:\nQ: {self.Dataset_class.get_single_question(self.data, idx)}\nA: <Prompt>\nOutput:\n{self.Dataset_class.get_single_answer(self.data, idx)}"
                for idx in self.format_index
            ]            
            task_example_string = "\n\n".join(task_example_list)
            instruction_pos_description = "at the beginning of the answer"
        else:
            if self.instruction_pos == 'Q_begin':
                task_example_list = [
                    f"Input:\n<Prompt>\n{self.Dataset_class.get_single_question(self.data, idx)}\nOutput:\n{self.Dataset_class.get_single_answer(self.data, idx)}"
                    for idx in self.format_index
                ]            
                task_example_string = "\n\n".join(task_example_list)
                instruction_pos_description = "at the beginning of the question"
            elif self.instruction_pos == 'Q_end':
                task_example_list = [
                    f"Input:\n{self.Dataset_class.get_single_question(self.data, idx)}\n<Prompt>\nOutput:\n{self.Dataset_class.get_single_answer(self.data, idx)}"
                    for idx in self.format_index
                ]            
                task_example_string = "\n\n".join(task_example_list)
                instruction_pos_description = "at the end of the question"

        if k_list_string is None and len(real_time_para_momentum) == 0:
            meta_util_gradient_text = (
                "Your task is to write a prompt to replace <Prompt>.\n\n"
                f"Below is the current prompt with its score."
                " The score ranges from 0 to 100, and higher score indicates better quality.\n"
                f"Prompt: {current_prompt}\nScore: {current_prompt_score}\n\n"
                f"The current prompt is:\n{current_prompt}\n\n"
                f"Below are the problems with this prompt.\n\n{gradient}\n\n"
                "The following exemplars show how to apply the prompt: you replace <Prompt> in each input with your new prompt, then read the input and give an output. We say your output is wrong if it is different from the given output, and we say your output is correct if they are the same.\n\n"
                f"{task_example_string}\n\n"
                "Write a new improved prompt"
                f" to replace <Prompt> {instruction_pos_description} in the task examples.\n"
                "Wrap the new prompt with <START> and <END>."
            )
        elif k_list_string is not None and len(real_time_para_momentum) == 0:
            meta_util_gradient_text = (
                "Your task is to write a prompt to replace <Prompt>.\n\n"
                f"{k_list_string}"
                f"The current prompt is:\n{current_prompt}\n\n"
                f"Below are the problems with this prompt.\n\n{gradient}\n\n"
                "The following exemplars show how to apply the prompt: you replace <Prompt> in each input with your new prompt, then read the input and give an output. We say your output is wrong if it is different from the given output, and we say your output is correct if they are the same.\n\n"
                f"{task_example_string}\n\n"
                "Carefully analyze the previous prompts and their scores,"
                " and write a new improved prompt"
                f" to replace <Prompt> {instruction_pos_description} in the task examples.\n"
                "Wrap the new prompt with <START> and <END>."
            )
        elif k_list_string is None and len(real_time_para_momentum) > 0:
            meta_util_gradient_text = (
                "Your task is to write a prompt to replace <Prompt>.\n\n"
                f"Below is the previous prompt with its score."
                " The score ranges from 0 to 100, and higher score indicates better quality.\n"
                f"Prompt: {real_time_para_momentum[0]}\nScore: {real_time_para_momentum[1]}\n\n"
                f"The current prompt is:\n{current_prompt}\n\n"
                f"Below are the problems with this prompt.\n\n{gradient}\n\n"
                "The following exemplars show how to apply the prompt: you replace <Prompt> in each input with your new prompt, then read the input and give an output. We say your output is wrong if it is different from the given output, and we say your output is correct if they are the same.\n\n"
                f"{task_example_string}\n\n"
                f"Carefully analyze the previous prompt and its score,"
                " and write a new improved prompt"
                f" to replace <Prompt> {instruction_pos_description} in the task examples.\n"
                "Wrap the new prompt with <START> and <END>."
            )
        
        return meta_util_gradient_text