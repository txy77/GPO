import os
import sys

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

class Linear_LR_Scheduler:
        
    def calculate_step_size(self, history_list, initial_step_size, use_warmup_strategy, warmup_steps, final_step_size, total_steps):
        decay_rate = (initial_step_size - final_step_size) / (total_steps - warmup_steps)
        current_step = len(history_list)
        if use_warmup_strategy:
            if current_step < warmup_steps:
                current_step_size = int(initial_step_size * current_step / warmup_steps)
            else:
                current_step_size = int(initial_step_size - decay_rate * (current_step - warmup_steps))
        else:
            current_step_size = int(initial_step_size - (decay_rate * current_step))
            
        return current_step_size