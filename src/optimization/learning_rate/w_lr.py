import os
import sys
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

lr_template = (
    "You are allowed to change up to {modify_word_num} words in the current prompt.\n"
)

class W_LR:
        
    def use_learning_rate(self, input_text, optimizer_llm_temperature_curr, num_generated_instructions_in_each_step, modify_word_num):
        input_text_prefix = '\n'.join(input_text.split("\n")[:-1])
        input_text_subfix = input_text.split("\n")[-1]
        lr_input_text = input_text_prefix + '\n' + lr_template.format(modify_word_num=modify_word_num) + '\n' + input_text_subfix
        input_text_list = [lr_input_text] * num_generated_instructions_in_each_step
        optimizer_llm_temperature_curr_list = [optimizer_llm_temperature_curr] * num_generated_instructions_in_each_step
        return input_text_list, optimizer_llm_temperature_curr_list