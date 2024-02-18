import os
import sys
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)


class Recency_Selection:
    def select(self, history_list, select_num, momentum_para_name):
        history_list = history_list[-select_num:]
        selected_list = history_list[::-1]
        if momentum_para_name in {'feedback'}:
            selected_list = [selected[0] for selected in selected_list]
        else:
            assert momentum_para_name == "para"
            selected_list = [(selected[0], selected[1]) for selected in selected_list] # instruction, score

        return selected_list
