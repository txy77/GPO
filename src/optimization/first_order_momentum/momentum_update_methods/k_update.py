import os
import sys
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)


class K_Update:
    def update_momentum(self, selected_list, momentum_para_name):
        
        if momentum_para_name in {'feedback'}:
            prefix_string = ""
            selected_string = "\n\n".join(selected_list)
        else:
            assert momentum_para_name == "para"
            prefix_string = (
                "Below are the previous prompts with their scores."
                " The score ranges from 0 to 100, and higher scores indicate better quality.\n\n"
            )
            selected_string = '\n\n'.join([f"Prompt: {selected_str[0]}\nScore: {selected_str[1]}" for selected_str in selected_list]) + '\n\n'
            
        complete_string = prefix_string + selected_string

        return complete_string
