import os
import sys
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)


class Importance_Selection:
    
    def select(self, history_list, select_num, momentum_para_name):
        if momentum_para_name in {'feedback'}:
            new = history_list[-1]
            history_list = sorted(history_list[:-1], key=lambda x: x[1], reverse=True)
            selected_list = history_list[:select_num]
            selected_list.insert(0, new)
            selected_list = [selected[0] for selected in selected_list] # instruction
        else:
            assert momentum_para_name == "para"
            history_list = sorted(history_list, key=lambda x: x[1], reverse=True)
            selected_list = history_list[:select_num]
            selected_list = [(selected[0], selected[1]) for selected in selected_list] # instruction, score
        
        return selected_list