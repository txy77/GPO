import os
import sys
import numpy as np

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

class Fixed_LR_Scheduler:
        
    def calculate_step_size(self, history_list, initial_step_size, use_warmup_strategy, warmup_steps):
        current_step = len(history_list)
        if use_warmup_strategy:
            if current_step < warmup_steps:
                current_step_size = int(initial_step_size * current_step / warmup_steps)
            else:
                current_step_size = initial_step_size
        else:
            current_step_size = initial_step_size
            
        return current_step_size