import os
import sys
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)


from src.dataset.base import Base_Dataset

class Real_time_Update:
    
    def update_momentum(self, new, real_time_momentum, momentum_para_name):
        
        if momentum_para_name in {"feedback"}:
            real_time_momentum_str = (
                "Below are the problems that arose from the previous prompts.\n"
                f"{real_time_momentum}\n\n"
            )
            momentum_string = (
                "Your task is to integrate the problems in the previous prompt and the current prompt.\n\n"
                f"{real_time_momentum_str}"
                "Below are the problems of the current prompt.\n"
                f"{new}\n\n"
                "You should integrate the problems of the previous prompt and the current prompt.\n"
                "Wrap the integrated problems with <START> and <END>."
            )
        else:
            assert momentum_para_name == "para"
            real_time_momentum_str = (
                "Below is the previous prompt with its score.\n"
                f"Prompt: {real_time_momentum[0]}\nScore: {real_time_momentum[1]}\n\n"
            )
            momentum_string = (
                "Your task is to summarize the previous and current prompt based on their scores and create a new prompt.\n"
                "The score ranges from 0 to 100, and higher score indicates better quality.\n\n"
                f"{real_time_momentum_str}"
                "Below is the current prompt with its score.\n"
                f"Prompt: {new[0]}\nScore: {new[1]}\n\n"
                "You should create a new prompt based on the previous and current prompt with their score.\n"
                "Wrap the created prompt with <START> and <END>."
            )

        return momentum_string
