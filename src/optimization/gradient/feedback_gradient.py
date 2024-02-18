import os
import sys
import numpy as np
import pandas as pd

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.dataset.base import Base_Dataset


class Feedback_Gradient:
    def __init__(self, dataset_name, data) -> None:
        self.Dataset_class = Base_Dataset(dataset_name)
        self.data = data

    def gen_gradient_text(self, current_prompt, train_batch_detailed_results_df, extract_final_answer_by_prompting_again):
        wrong_question_indices_list = list(
            set(
                list(
                    train_batch_detailed_results_df.iloc[
                        np.where(train_batch_detailed_results_df.accuracy == 0.0)[0],
                        :,
                    ].index
                )
            )
        )

        zero_accuracy_rows = train_batch_detailed_results_df[
            train_batch_detailed_results_df["accuracy"] == 0.0
        ]
        
        try:  
            raw_index2prediction = zero_accuracy_rows["raw_answer_second_round"].to_dict()
        except:
            raw_index2prediction = zero_accuracy_rows["parsed_answer"].to_dict()

        error_list = [
            f"Question: {self.Dataset_class.get_single_question(self.data, idx)}\n"
            f"Wrong prediction: {raw_index2prediction[idx]}\n"
            f"Ground truth answer: {self.Dataset_class.get_single_answer(self.data, idx)}"
            for idx in wrong_question_indices_list
        ]
        error_string = "\n\n".join(error_list)

        meta_gradient_text = (
            "Your task is to point out the problems with the current prompt based on the wrong examples.\n\n"
            f"The current prompt is:\n{current_prompt}\n\n"
            f"But this prompt gets the following examples wrong.\n"
            "You should analyze the differences between wrong predictions and ground truth answers,"
            " and carefully consider why this prompt led to incorrect predictions.\n\n"
            f"Below are the task examples with Queston, Wrong prediction, and Ground truth answer.\n\n{error_string}\n\n"
            "Give a reason why the prompt could have gotten these examples wrong.\n"
            "Wrap the reason with <START> and <END>."
        )

        return meta_gradient_text
