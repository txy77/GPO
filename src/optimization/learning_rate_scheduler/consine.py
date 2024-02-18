import os
import sys
import numpy as np

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

class Consine_LR_Scheduler:
        
    def calculate_step_size(self, history_list, initial_step_size, use_warmup_strategy, warmup_steps, final_step_size, total_steps):
        current_step = len(history_list)
        if use_warmup_strategy:
            if len(history_list) < warmup_steps:
                current_step_size = int(initial_step_size * current_step / warmup_steps)
            else:
                decayed = (current_step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * decayed))
                current_step_size = int(final_step_size + (initial_step_size - final_step_size) * cosine_decay)
        else:
            decayed = current_step / total_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decayed))
            current_step_size = int(final_step_size + (initial_step_size - final_step_size) * cosine_decay)
        return current_step_size