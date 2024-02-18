import os
import sys
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

class Wo_LR:
        
    def use_learning_rate(self, input_text, optimizer_llm_temperature_curr, num_generated_instructions_in_each_step):
        input_text_list = [input_text] * num_generated_instructions_in_each_step
        optimizer_llm_temperature_curr_list = [optimizer_llm_temperature_curr] * num_generated_instructions_in_each_step
        return input_text_list, optimizer_llm_temperature_curr_list