"""The .py file for prompt optimization"""

import collections
import json
import os
import pickle
import random
import re
import sys
import openai
import functools
import numpy as np
import pandas as pd
import pdb
from copy import deepcopy

from nltk.tokenize import word_tokenize

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.dataset.base import Base_Dataset
from src import prompt_utils
from src.evaluation import eval_utils
from src.optimization.gradient.base_gradient import Base_Gradient
from src.optimization.learning_rate.base_lr import Base_LR
from src.optimization.utilize_gradient.base_util_gradient import Base_Util_Gradient
from src.optimization.first_order_momentum.base_fo_momentum import (
    Base_FO_Momemtum,
)
from src.optimization.learning_rate_scheduler.base_lr_scheduler import (
    Base_LR_Scheduler,
)


# extract text from raw string wrapped with start_string and end_string
def extract_text(raw_string, start_string="<START>", end_string="<END>"):
    if start_string not in raw_string:
        start_index = 0
    else:
        start_index = raw_string.index(start_string) + len(start_string)
    if end_string not in raw_string:
        end_index = len(raw_string)
    else:
        end_index = raw_string.index(end_string)
    extracted_string = raw_string[start_index:end_index].strip()
    return extracted_string


def random_sampling(sample_data_from, all_num_examples, ratio):
    index = np.sort(
        np.array(
            np.random.choice(
                sample_data_from, size=int(ratio * all_num_examples), replace=False
            )
        )
    )
    return index

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def prompt_optimization(
    openai_main_api_key,
    openai_api_key_list,
    gpus,
    dataset_name,
    task_name,
    optimizer_llm_name,
    optimizer_temperature,
    initial_instruction,
    instruction_pos,
    opt_batch_size,
    format_data_num,
    gradient_name,
    momentum_para_name,
    momentum_selection_name,
    momentum_selection_num,
    momentum_update_name,
    learning_rate_name,
    initial_step_size,
    decay_strategy,
    use_warmup_strategy,
    warmup_steps,
    final_step_size,
    util_gradient_name,
    num_search_epochs,
    num_generated_instructions_in_each_step,
    scorer_llm_name,
    include_qa,
    evaluate_generated_ins_on_few_shot,
    few_shot_num,
    result_by_instruction_folder,
    save_folder,
    only_evaluate,
):
    # ====================== environment setting ======================
    np.random.seed(0)
    random.seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    # ====================== optimizer model config ======================
    assert optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
    openai.api_key = openai_main_api_key
    optimizer_gpt_max_decode_steps = 512
    optimizer_gpt_temperature = optimizer_temperature

    optimizer_llm_dict = dict()
    optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
    optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
    optimizer_llm_dict["model_type"] = optimizer_llm_name.lower()
    call_optimizer_server_func = functools.partial(
        prompt_utils.call_openai_server_func,
        model=optimizer_llm_name,
        n=num_generated_instructions_in_each_step,
        max_decode_steps=optimizer_gpt_max_decode_steps,
        temperature=optimizer_gpt_temperature,
    )

    # ====================== scorer model config ======================
    if scorer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
        openai.api_key = openai_main_api_key
        scorer_gpt_max_decode_steps = 256
        scorer_gpt_temperature = 0.0

        scorer_gpt_dict = dict()
        scorer_gpt_dict["max_decode_steps"] = scorer_gpt_max_decode_steps
        scorer_gpt_dict["temperature"] = scorer_gpt_temperature

        scorer_llm_dict = {
            "model_type": scorer_llm_name.lower(),
        }
        tokenizer = None
        scorer_llm_dict.update(scorer_gpt_dict)
        call_scorer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=scorer_llm_name.lower(),
            n=1,
            max_decode_steps=scorer_gpt_max_decode_steps,
            temperature=scorer_gpt_temperature,
        )
    else:
        assert scorer_llm_name in {
            "llama2-7b",
            "llama2-chat-7b",
            "llama2-chat-13b",
        }
        scorer_vllm_max_decode_steps = 256
        scorer_vllm_temperature = 0.0

        scorer_vllm_dict = dict()
        scorer_vllm_dict["max_decode_steps"] = scorer_vllm_max_decode_steps
        scorer_vllm_dict["temperature"] = scorer_vllm_temperature

        scorer_llm_dict = {
            "model_type": scorer_llm_name.lower(),
        }
        scorer_llm_dict.update(scorer_vllm_dict)
        model_path = prompt_utils.model2model_path[scorer_llm_name]
        model = prompt_utils.load_vllm_llm(model_path, tensor_parallel_size=len(gpus))
        tokenizer = model.get_tokenizer()
        call_scorer_server_func = functools.partial(
            prompt_utils.call_vllm_server_func,
            model=model,
            llm_name=scorer_llm_name,
            max_decode_steps=scorer_vllm_max_decode_steps,
            temperature=scorer_vllm_temperature,
        )

    # ====================== try calling the model ======================
    print("\n======== testing the scorer and optimizer servers ===========")

    optimizer_test_output = call_optimizer_server_func(
        "Does the sun rise from the north? Just answer yes or no.",
        temperature=optimizer_temperature,
    )
    print(f"number of optimizer output decodes: {len(optimizer_test_output)}")
    print(f"optimizer test output: {optimizer_test_output}")
    print("Finished testing the optimizer servers.")

    scorer_test_output = call_scorer_server_func(
        "Does the sun rise from the north? Just answer yes or no."
    )
    print(f"number of scorer output decodes: {len(scorer_test_output)}")
    print(f"scorer test output: {scorer_test_output}")
    print("Finished testing the scorer servers.")
    
    # =================== save configurations to json file ====================
    configs_dict = dict()
    configs_dict["dataset_name"] = dataset_name
    configs_dict["task_name"] = task_name
    configs_dict["initial_instruction"] = initial_instruction
    configs_dict["num_search_epochs"] = num_search_epochs
    configs_dict["opt_batch_size"] = opt_batch_size
    configs_dict["num_search_epochs"] = num_search_epochs
    configs_dict[
        "num_generated_instructions_in_each_step"
    ] = num_generated_instructions_in_each_step
    configs_dict["format_data_num"] = format_data_num

    configs_dict["gradient_name"] = gradient_name
    configs_dict["momentum_para_name"] = momentum_para_name
    configs_dict["momentum_selection_name"] = momentum_selection_name
    configs_dict["momentum_select_num"] = momentum_selection_num
    configs_dict["momentum_update_name"] = momentum_update_name
    configs_dict["learning_rate_name"] = learning_rate_name
    configs_dict["initial_step_size"] = initial_step_size
    configs_dict["decay_strategy"] = decay_strategy
    configs_dict["use_warmup_strategy"] = use_warmup_strategy
    configs_dict["warmup_steps"] = warmup_steps
    configs_dict["final_step_size"] = final_step_size
    
    configs_dict["util_gradient_name"] = util_gradient_name

    configs_dict["scorer_llm_dict"] = scorer_llm_dict
    configs_dict["optimizer_llm_dict"] = optimizer_llm_dict
    configs_dict["instruction_pos"] = instruction_pos
    configs_dict["include_qa"] = include_qa
    
    configs_dict["evaluate_generated_ins_on_few_shot"] = evaluate_generated_ins_on_few_shot
    configs_dict["few_shot_num"] = few_shot_num
    
    with open(os.path.join(save_folder, "configs_dict.json"), "w") as f:
        json.dump(configs_dict, f, indent=4)

    # ====================== load dataset, dataset setting and dataset ratio ======================

    Dataset_class = Base_Dataset(dataset_name)
    data_list = Dataset_class.read_data(task_name)
    train_ratio, eval_ratio, test_ratio = Dataset_class.get_ratio()

    # ====================== sample data ======================
    evaluating_datas = []  # Data prepared for evaluation
    for data in data_list:
        # task
        task = data["task"]
        setting_dict = Dataset_class.get_task_setting(task)
        is_multiple_choice = setting_dict["is_multiple_choice"]
        prediction_treat_as_number = setting_dict["prediction_treat_as_number"]
        prediction_treat_as_bool = setting_dict["prediction_treat_as_bool"]
        prediction_treat_as_rouge = setting_dict["prediction_treat_as_rouge"]
        extract_final_answer_by_prompting_again = setting_dict["extract_final_answer_by_prompting_again"]

        # sample train data
        train_data = data["train_data"]
        train_num_examples = data["train_num_examples"]
        train_index = random_sampling(train_num_examples, train_num_examples, train_ratio)

        # sample eval data
        eval_data = data["eval_data"]
        eval_num_examples = data["eval_num_examples"]
        eval_index = random_sampling(eval_num_examples, eval_num_examples, eval_ratio)

        # sample test data
        test_data = data["test_data"]
        test_num_examples = data["test_num_examples"]
        test_index = random_sampling(test_num_examples, test_num_examples, test_ratio)

        # sample format data
        format_data = data["format_data"]
        format_num_examples = data["format_num_examples"]
        format_index = np.sort(
            np.array(
                np.random.choice(
                    format_num_examples, size=format_data_num, replace=False
                )
            )
        )

        few_shot_data = data["few_shot_data"]
        few_shot_num_examples = data["few_shot_num_examples"]

        train_sample_num, eval_sample_num, test_sample_num, format_sample_num = (
            train_index.shape[0],
            eval_index.shape[0],
            test_index.shape[0],
            format_index.shape[0],
        )

        evaluating_datas.append(
            {
                "task": deepcopy(task),
                "train_data": train_data,
                "train_index": train_index,
                "eval_data": eval_data,
                "eval_index": eval_index,
                "test_data": test_data,
                "test_index": test_index,
                "format_data": format_data,
                "format_index": format_index,
                "few_shot_data": few_shot_data,
                "is_multiple_choice": is_multiple_choice,
                "prediction_treat_as_number": prediction_treat_as_number,
                "prediction_treat_as_bool": prediction_treat_as_bool,
                "prediction_treat_as_rouge": prediction_treat_as_rouge,
                "extract_final_answer_by_prompting_again": extract_final_answer_by_prompting_again
            }
        )

        print(f"Finish sample data from {dataset_name} {task_name}")
        print(
            f"train sample num: {train_sample_num}, eval sample num: {eval_sample_num}, test sample num: {test_sample_num}, format_num: {format_sample_num}"
        )
    
    for evaluating_data in evaluating_datas:        
        
        # ====================== Initiate parameters ======================

        # format: [(instruction, score, epoch_index, step_index)] step_index = -1 means initial prompt
        old_instructions_and_scores_eval = []
        # [instruction(str), score(float), the iteration step epoch(int), the iteration step index(int)]
        meta_gradient_prompts = []
        # format: [(meta_gradient_prompt, epoch_index, step_index)]
        meta_gradient_texts = (
            []
        )  # format: [(meta_gradient_text, improved_score, epoch_index, step_index)]
        # meta_prompst: Meta-prompt generated at each iteration step

        selected_gradient_history_list = []
        meta_fo_momentum_gradient_prompts = []
        meta_fo_momentum_gradient_texts = []
        
        selected_para_history_list = []
        meta_fo_momentum_para_prompts = []
        meta_fo_momentum_para_texts = []

        meta_prompts = []  # format: [(meta_prompt, epoch_index, step_index)]
        meta_generate_instructions = (
            []
        )  # format: [(meta_generate_instructions(list), epoch_index, step_index)]
        # step_index: The index or number of the step iteration
        instruction_score_dict = dict()  # the dictionary of {instruction: score}
        # the dictionary of the few-shot QA indices in meta-prompt
        # key: step index; value: the list of few-shot indices in that step
        train_detailed_results_df_by_instruction_dict = (
            dict()
        )  # record instruction performance on train set
        train_log = collections.defaultdict(
            list
        )  # key: instruction, value: [score, all_num, epoch_index, step_index]
        eval_detailed_results_df_by_instruction_dict = (
            dict()
        )  # record instruction performance on eval set
        eval_log = collections.defaultdict(
            list
        )  # key: instruction, value: [score, epoch_index, step_index]
        test_detailed_results_df_by_instruction_dict = (
            dict()
        )  # record instruction performance on test set
        test_log = collections.defaultdict(
            list
        )  # key: instruction, value: [score, epoch_index, step_index]
        eval_step2max_score = dict()
        eval_step2max_instruction = dict()

        old_instruction_md5_hashstrings_set = (
            set()
        )  # md5 hashstrings set of old instruction
        prev_saved_instructions = set() # record instructions
        
        modify_word_num_history = [] # record modified word num history (modify_word_num, epoch, step)
        
        # ====================== load task-related data ======================
        task = evaluating_data["task"]
        # initial instruction
        if dataset_name == "mmlu" and initial_instruction.startswith("The following are multiple choice questions about"):
            initial_instruction = "The following are multiple choice questions about" + format_subject(task) + '.'
        train_data = evaluating_data["train_data"]
        train_index = evaluating_data["train_index"]
        eval_data = evaluating_data["eval_data"]
        eval_index = evaluating_data["eval_index"]
        test_data = evaluating_data["test_data"]
        test_index = evaluating_data["test_index"]
        format_data = evaluating_data["format_data"]
        format_index = evaluating_data["format_index"]
        few_shot_data = evaluating_data["few_shot_data"]
        is_multiple_choice = evaluating_data["is_multiple_choice"]
        prediction_treat_as_number = evaluating_data["prediction_treat_as_number"]
        prediction_treat_as_bool = evaluating_data["prediction_treat_as_bool"]
        prediction_treat_as_rouge = evaluating_data["prediction_treat_as_rouge"]
        extract_final_answer_by_prompting_again = evaluating_data["extract_final_answer_by_prompting_again"]
        
        setting_dict_for_save = {
            "is_multiple_choice": is_multiple_choice,
            "prediction_treat_as_number": prediction_treat_as_number,
            "prediction_treat_as_bool": prediction_treat_as_bool,
            "prediction_treat_as_rouge": prediction_treat_as_rouge,
            "extract_final_answer_by_prompting_again": extract_final_answer_by_prompting_again,
        }
        
        # ====================== create save directory ======================
        train_result_by_instruction_folder = (
            f"{result_by_instruction_folder}/{task}/train"
        )
        eval_result_by_instruction_folder = (
            f"{result_by_instruction_folder}/{task}/eval"
        )
        test_result_by_instruction_folder = (
            f"{result_by_instruction_folder}/{task}/test"
        )

        os.makedirs(train_result_by_instruction_folder, exist_ok=True)
        os.makedirs(eval_result_by_instruction_folder, exist_ok=True)
        os.makedirs(test_result_by_instruction_folder, exist_ok=True)
        
        # ====================== evaluate initial instructions =====================
        print("\n============== evaluating initial instructions ===============")
        print(f"""computing the score of "{initial_instruction}" by prompting""")

        if not only_evaluate:        
            eval_initial_detailed_results_df = eval_utils.evaluate_single_instruction(
                data=eval_data,
                scorer_llm_name=scorer_llm_name,
                instruction=initial_instruction,
                eval_index_all=eval_index,
                tokenizer=tokenizer,
                call_server_func=call_scorer_server_func,
                dataset_name=dataset_name,
                task_name=task,
                extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
                include_qa=include_qa,
                instruction_pos=instruction_pos,
                is_multiple_choice=is_multiple_choice,
                evaluate_generated_ins_on_few_shot=evaluate_generated_ins_on_few_shot,
                few_shot_data=few_shot_data,
                few_shot_num=few_shot_num,
                prediction_treat_as_number=prediction_treat_as_number,
                prediction_treat_as_bool=prediction_treat_as_bool,
                prediction_treat_as_rouge=prediction_treat_as_rouge,
                prediction_num_decimals=0,
            )
            eval_detailed_results_df_by_instruction_dict[
                initial_instruction
            ] = eval_initial_detailed_results_df
            eval_scores = eval_initial_detailed_results_df["accuracy"]
            eval_average_score = np.average(eval_scores)
            print(f"instruction: {initial_instruction}, score: {eval_average_score}")

            eval_log["-1_-1"] = (initial_instruction, eval_average_score, -1, -1)
            eval_step2max_score[-1] = eval_average_score
            eval_step2max_instruction[-1] = eval_average_score

            eval_filename = eval_utils.instruction_to_filename(initial_instruction)
            eval_epoch_step_folder = os.path.join(
                eval_result_by_instruction_folder, str(-1), str(-1)
            )
            os.makedirs(eval_epoch_step_folder, exist_ok=True)
            eval_file_path = os.path.join(eval_epoch_step_folder, f"{eval_filename}.csv")
            eval_initial_detailed_results_df.to_csv(eval_file_path, index=True, header=True)
            print(f"""saving results of "{initial_instruction}" to {eval_file_path}""")
            old_instructions_and_scores_eval.append(
                (initial_instruction, eval_average_score, -1, -1)
            )
            instruction_score_dict[initial_instruction] = eval_average_score

        test_initial_detailed_results_df = eval_utils.evaluate_single_instruction(
            data=test_data,
            scorer_llm_name=scorer_llm_name,
            instruction=initial_instruction,
            eval_index_all=test_index,
            tokenizer=tokenizer,
            call_server_func=call_scorer_server_func,
            dataset_name=dataset_name,
            task_name=task,
            extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
            include_qa=include_qa,
            instruction_pos=instruction_pos,
            is_multiple_choice=is_multiple_choice,
            evaluate_generated_ins_on_few_shot=evaluate_generated_ins_on_few_shot,
            few_shot_data=few_shot_data,
            few_shot_num=few_shot_num,
            prediction_treat_as_number=prediction_treat_as_number,
            prediction_treat_as_bool=prediction_treat_as_bool,
            prediction_treat_as_rouge=prediction_treat_as_rouge,
            prediction_num_decimals=0,
        )

        test_detailed_results_df_by_instruction_dict[
            initial_instruction
        ] = test_initial_detailed_results_df
        test_scores = test_initial_detailed_results_df["accuracy"]
        test_average_score = np.average(test_scores)
        print(f"instruction: {initial_instruction}, score: {test_average_score}")

        test_log["-1_-1"] = (initial_instruction, test_average_score, -1, -1)

        test_filename = eval_utils.instruction_to_filename(initial_instruction)
        test_epoch_step_folder = os.path.join(
            test_result_by_instruction_folder, str(-1), str(-1)
        )
        os.makedirs(test_epoch_step_folder, exist_ok=True)
        test_file_path = os.path.join(test_epoch_step_folder, f"{test_filename}.csv")
        test_initial_detailed_results_df.to_csv(test_file_path, index=True, header=True)
        print(f"""saving results of "{initial_instruction}" to {test_file_path}""")

        if only_evaluate:
            # ===================== save up-to-date results ===========================
            results_dict = dict()
            results_dict["test_log"] = test_log
            results_dict["setting_dict"] = setting_dict_for_save

            # add result eval test
            with open(os.path.join(result_by_instruction_folder, task, "results_dict.json"), "w") as fp:
                json.dump(results_dict, fp)
            print(f"\nsaved all results to\n{save_folder}")
            continue
        # ================== evolution ==================

        # ================== Define evolution class ==================
        print("\n============== evolution instructions ===============")
        if gradient_name in {"feedback"}:
            Gradient_class = Base_Gradient(
                gradient_name,
                dataset_name=dataset_name,
                data=train_data,
            )
            
        if (momentum_selection_name in {
            "recency",
            "relavance",
            "importance",
        } and momentum_update_name == "k-list") or momentum_update_name == "real-time":
            First_order_momentum_class = Base_FO_Momemtum(
                momentum_selection_name, momentum_update_name
            )
        
        LR_class = Base_LR(learning_rate_name)
        
        if learning_rate_name == "w_lr" and decay_strategy in {"fixed", "linear", "consine"}:
            lr_scheduler = Base_LR_Scheduler(decay_strategy)

        Util_gradient_class = Base_Util_Gradient(
            util_gradient_name,
            dataset_name=dataset_name,
            data=format_data,
            instruction_pos=instruction_pos,
            include_qa=include_qa,
            format_index=format_index,
        )
            
        update_instruction = initial_instruction
        real_time_gradient_momentum = ""  # use for real-time update gradient momentum
        real_time_para_momentum = [] # use for real-time update para momentum, (prompt, score)
        
        last_round_num = train_sample_num % opt_batch_size
        if last_round_num < (opt_batch_size / 2):
            steps = train_sample_num // opt_batch_size
        else:
            steps = train_sample_num // opt_batch_size + 1
        total_step_size = num_search_epochs * steps

        for i_epoch in range(num_search_epochs):
            print(f"\n================== Epoch {i_epoch} =====================")
            optimizer_llm_temperature_curr = optimizer_llm_dict["temperature"]
            
            for i_step in range(steps):
                print(f"================== step {i_step} =====================")
                train_sample_index = train_index
                train_sample_index = train_index[
                    i_step
                    * opt_batch_size : min(
                        train_sample_num, (i_step + 1) * opt_batch_size
                    )
                ]
                train_batch_detailed_results_df = eval_utils.evaluate_single_instruction(
                    data=train_data,
                    scorer_llm_name=scorer_llm_name,
                    instruction=update_instruction,
                    eval_index_all=train_sample_index,
                    tokenizer=tokenizer,
                    call_server_func=call_scorer_server_func,
                    dataset_name=dataset_name,
                    task_name=task,
                    extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
                    include_qa=include_qa,
                    instruction_pos=instruction_pos,
                    is_multiple_choice=is_multiple_choice,
                    evaluate_generated_ins_on_few_shot=evaluate_generated_ins_on_few_shot,
                    few_shot_data=few_shot_data,
                    few_shot_num=few_shot_num,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_treat_as_rouge=prediction_treat_as_rouge,
                    prediction_num_decimals=0,
                )

                train_scores = train_batch_detailed_results_df["accuracy"]
                train_average_score = np.average(train_scores)
                train_log[f"{i_epoch}_{i_step}"] = (
                    update_instruction,
                    train_average_score,
                    i_epoch,
                    i_step,
                )
                train_filename = eval_utils.instruction_to_filename(update_instruction)
                train_epoch_step_folder = os.path.join(
                    train_result_by_instruction_folder, str(i_epoch), str(i_step)
                )
                os.makedirs(train_epoch_step_folder, exist_ok=True)
                train_file_path = os.path.join(
                    train_epoch_step_folder, f"""{train_filename}.csv"""
                )
                train_batch_detailed_results_df.to_csv(
                    train_file_path, index=True, header=True
                )
                print(f"saving results to {train_file_path}")

                train_detailed_results_df_by_instruction_dict[
                    update_instruction
                ] = train_batch_detailed_results_df

                # ================== generate gradient ==================
                if gradient_name in {"feedback"}:
                    if gradient_name == "feedback":
                        meta_gradient_prompt = Gradient_class.gen_gradient_text(
                            update_instruction, train_batch_detailed_results_df, extract_final_answer_by_prompting_again
                        )
                    # print(f"meta_gradient_prompt: \n\n{meta_gradient_prompt}\n")
                    meta_gradient_prompts.append(
                        (meta_gradient_prompt, i_epoch, i_step)
                    )

                    optimizer_llm_gradient_temperature_curr = (
                        0.0  # only generate 1 result(gradient)
                    )
                    gradient_text = call_optimizer_server_func(
                        meta_gradient_prompt,
                        temperature=optimizer_llm_gradient_temperature_curr,
                    )

                    raw_result = gradient_text[0]  # only one result
                    new_gradient = extract_text(
                        raw_result, start_string="<START>", end_string="<END>"
                    )
                    new_gradient = eval_utils.polish_sentence(new_gradient)
                    meta_gradient_texts.append([new_gradient, -100, i_epoch, i_step])
                    print(f"\ninitially generated gradient.\n")
                    
                    # ================== generate gradients momentum ==================
                    if momentum_para_name in {"feedback"}:
                        if momentum_update_name == "k-list":
                            assert momentum_selection_name in {"recency", "relavance", "importance"}
                            selected_gradient_list = (
                                First_order_momentum_class.select(
                                    meta_gradient_texts, momentum_selection_num, momentum_para_name
                                )
                            )

                            selected_gradient_history_list.append(
                                (selected_gradient_list, i_epoch, i_step)
                            )
                            new_gradient = First_order_momentum_class.update_momentum(
                                selected_gradient_list,
                                momentum_para_name,
                            )

                        else:
                            assert momentum_update_name == "real-time"
                            if len(real_time_gradient_momentum) > 0:
                                momentum_prompt = First_order_momentum_class.update_momentum(
                                    new_gradient, real_time_gradient_momentum, momentum_para_name
                                )
                                meta_fo_momentum_gradient_prompts.append(
                                    (momentum_prompt, i_epoch, i_step)
                                )

                                optimizer_llm_real_time_momentum_temperature_curr = 0.0
                                momentum_gradient = call_optimizer_server_func(
                                    momentum_prompt,
                                    temperature=optimizer_llm_real_time_momentum_temperature_curr,
                                )
                                momentum_gradient = extract_text(
                                    momentum_gradient[0], start_string="<START>", end_string="<END>"
                                )
                                momentum_gradient = eval_utils.polish_sentence(momentum_gradient)
                                meta_fo_momentum_gradient_texts.append((momentum_gradient, i_epoch, i_step))
                                real_time_gradient_momentum = momentum_gradient
                                new_gradient = momentum_gradient
                            else:
                                real_time_gradient_momentum = new_gradient
                        print("\ninitially generated gradient momentum.\n")

                # ================== generate parameters momentum ==================
                k_list_string = None
                if momentum_para_name == "para":
                    if momentum_update_name == "k-list":
                        assert momentum_selection_name in {"recency", "relavance", "importance"}
                        selected_para_list = First_order_momentum_class.select(
                            old_instructions_and_scores_eval, momentum_selection_num, momentum_para_name
                        )
                        selected_para_history_list.append(
                            (selected_para_list, i_epoch, i_step)
                        )
                        
                        k_list_string = (
                            First_order_momentum_class.update_momentum(
                                selected_para_list,
                                momentum_para_name,
                            )
                        )
                    else:
                        assert momentum_update_name == "real-time"
                        update_instruction_dict = (update_instruction, round(instruction_score_dict[update_instruction], 2))
                        if len(real_time_para_momentum) > 0:
                            momentum_prompt = First_order_momentum_class.update_momentum(
                                update_instruction_dict, real_time_para_momentum, momentum_para_name
                            )
                            meta_fo_momentum_para_prompts.append(
                                (momentum_prompt, i_epoch, i_step)
                            )
                            optimizer_llm_momentum_temperature_curr = 0.0
                            momentum_para = call_optimizer_server_func(
                                momentum_prompt,
                                temperature=optimizer_llm_momentum_temperature_curr,
                            )
                            real_time_para_string = extract_text(
                                momentum_para[0], start_string="<START>", end_string="<END>"
                            )
                            real_time_para_string = eval_utils.polish_sentence(real_time_para_string)
                            
                            eval_summarized_prompt_detailed_results_df = eval_utils.evaluate_single_instruction(
                                scorer_llm_name=scorer_llm_name,
                                instruction=update_instruction,
                                data=eval_data,
                                eval_index_all=eval_index,
                                tokenizer=tokenizer,
                                call_server_func=call_scorer_server_func,
                                dataset_name=dataset_name,
                                task_name=task,
                                extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
                                include_qa=include_qa,
                                instruction_pos=instruction_pos,
                                is_multiple_choice=is_multiple_choice,
                                evaluate_generated_ins_on_few_shot=evaluate_generated_ins_on_few_shot,
                                few_shot_data=few_shot_data,
                                few_shot_num=few_shot_num,
                                prediction_treat_as_number=prediction_treat_as_number,
                                prediction_treat_as_bool=prediction_treat_as_bool,
                                prediction_treat_as_rouge=prediction_treat_as_rouge,
                                prediction_num_decimals=0,
                            )

                            eval_summarized_prompt_scores = eval_summarized_prompt_detailed_results_df["accuracy"]
                            eval_summarized_prompt_average_score = np.average(eval_summarized_prompt_scores)
                            
                            meta_fo_momentum_para_texts.append((real_time_para_string, eval_summarized_prompt_average_score, i_epoch, i_step))
                            
                            real_time_para_momentum = [real_time_para_string, round(eval_summarized_prompt_average_score, 2)]
                            
                        else:
                            real_time_para_momentum = [update_instruction, round(instruction_score_dict[update_instruction], 2)]
                        
                        print("\ninitially generated parameter momentum.\n")
                
                # ================== use LR scheduler ==================
                if decay_strategy in {"fixed", "linear", "consine"}:
                    if decay_strategy == "fixed":
                        current_step_size = lr_scheduler.calculate_step_size(old_instructions_and_scores_eval, initial_step_size, use_warmup_strategy, warmup_steps)
                    else:
                        assert decay_strategy in {"linear", "consine"}
                        current_step_size = lr_scheduler.calculate_step_size(old_instructions_and_scores_eval, initial_step_size, use_warmup_strategy, warmup_steps, final_step_size, total_step_size)

                # ================== generate text utilizing the gradient ==================
                update_instruction_score = instruction_score_dict[update_instruction]
                update_instruction_score = round(update_instruction_score, 2)
                if util_gradient_name in {"generate", "edit"}:
                    meta_util_gradient_prompt = (
                        Util_gradient_class.gen_util_gradient_text(
                            update_instruction,
                            update_instruction_score,
                            new_gradient,
                            k_list_string,
                            real_time_para_momentum,
                        )
                    )
                else:
                    assert util_gradient_name in {"generate_without", "edit_without"}
                    meta_util_gradient_prompt = (
                        Util_gradient_class.gen_util_gradient_text(
                            update_instruction,
                            update_instruction_score,
                            k_list_string,
                            real_time_para_momentum,
                        )
                    )

                generated_instructions_raw = []
                
                # pdb.set_trace()

                # ================== use learning rate ==================
                if learning_rate_name == "wo_lr":
                    (
                        optimizer_llm_input_text_list,
                        optimizer_llm_temperature_curr_list,
                    ) = LR_class.use_learning_rate(
                        meta_util_gradient_prompt, optimizer_llm_temperature_curr, num_generated_instructions_in_each_step
                    )
                else:
                    assert learning_rate_name == "w_lr"
                    (
                        optimizer_llm_input_text_list,
                        optimizer_llm_temperature_curr_list,
                    ) = LR_class.use_learning_rate(
                        meta_util_gradient_prompt, optimizer_llm_temperature_curr, num_generated_instructions_in_each_step, current_step_size
                    )
                    modify_word_num_history.append((current_step_size, i_epoch, i_step))
                
                meta_prompts.append((optimizer_llm_input_text_list[0], i_epoch, i_step))
                
                for optimizer_llm_input_text, optimizer_llm_temperature_curr in zip(
                    optimizer_llm_input_text_list, optimizer_llm_temperature_curr_list
                ):
                    # generate instructions
                    raw_outputs = call_optimizer_server_func(
                        optimizer_llm_input_text,
                        temperature=optimizer_llm_temperature_curr,
                    )
                    raw_output = raw_outputs[0]
                    new_inst = extract_text(
                        raw_output, start_string="<START>", end_string="<END>"
                    )
                    generated_instructions_raw.append(new_inst)

                generated_instructions_raw = list(
                    map(eval_utils.polish_sentence, generated_instructions_raw)
                )
                print(f"\ninitially generated prompts: {generated_instructions_raw}\n")
                # pdb.set_trace()

                # ================== evaluate generated instructions ==================
                print(
                    "\n============== evaluating newly generated instructions on the eval and testing set ==============="
                )
                current_instruction2score = (
                    dict()
                )  # record performance of current instruction
                current_instruction2score[update_instruction] = 0.0 # add last round instruction to current_instruction2score list
                eval_cnt = 0
                generated_instructions = (
                    []
                )  # the new instructions generated in this step
                for ins in generated_instructions_raw:
                    ins_md5_hashstring = eval_utils.instruction_to_filename(
                        ins, md5_hashing=True
                    )
                    if ins_md5_hashstring in old_instruction_md5_hashstrings_set:
                        # print(f"already evaluated '{ins}' previously")
                        if ins in instruction_score_dict:
                            eval_average_score = instruction_score_dict[ins]
                            eval_log[f"{i_epoch}_{i_step}_{eval_cnt}"] = (
                                instruction,
                                eval_average_score,
                                i_epoch,
                                i_step,
                            )
                            eval_cnt += 1
                            # if ins not in current_instruction2score:
                            current_instruction2score[ins] = eval_average_score
                    else:
                        generated_instructions.append(ins)
                        old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)
                generated_instructions = list(set(generated_instructions))

                # ================== rule-based filtering ==================
                to_evaluate_instructions = []

                for instruction in generated_instructions:
                    to_evaluate_instructions.append(instruction)

                print(f"len(to_evaluate_instructions), {len(to_evaluate_instructions)}")
                meta_generate_instructions.append(
                    (to_evaluate_instructions, i_epoch, i_step)
                )

                for instruction in to_evaluate_instructions:
                    if instruction not in prev_saved_instructions:
                        print(
                            f"""computing the score of "{instruction}" by prompting"""
                        )
                        prev_saved_instructions.add(instruction)

                eval_detailed_results_dict = eval_utils.evaluate_parallel_instructions(
                    api_keys=openai_api_key_list,
                    scorer_llm_name=scorer_llm_name,
                    instructions=to_evaluate_instructions,
                    data=eval_data,
                    eval_index_all=eval_index,
                    tokenizer=tokenizer,
                    call_server_func=call_scorer_server_func,
                    dataset_name=dataset_name,
                    task_name=task,
                    extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
                    include_qa=include_qa,
                    instruction_pos=instruction_pos,
                    is_multiple_choice=is_multiple_choice,
                    evaluate_generated_ins_on_few_shot=evaluate_generated_ins_on_few_shot,
                    few_shot_data=few_shot_data,
                    few_shot_num=few_shot_num,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_treat_as_rouge=prediction_treat_as_rouge,
                    prediction_num_decimals=0,
                )

                for eval_idx, (
                    instruction,
                    eval_detailed_results_df,
                ) in enumerate(eval_detailed_results_dict.items()):
                    eval_scores = eval_detailed_results_df["accuracy"]
                    eval_average_score = np.average(eval_scores)
                    current_instruction2score[instruction] = float(eval_average_score)
                    print(
                        f"Epoch {i_epoch}, Step {i_step}, instruction: {instruction}, score: {eval_average_score}"
                    )

                    eval_epoch_step_folder = os.path.join(
                        eval_result_by_instruction_folder, str(i_epoch), str(i_step)
                    )
                    os.makedirs(eval_epoch_step_folder, exist_ok=True)
                    eval_filename = eval_utils.instruction_to_filename(instruction)
                    eval_file_path = os.path.join(
                        eval_epoch_step_folder, f"""{eval_filename}.csv"""
                    )
                    eval_detailed_results_df.to_csv(
                        eval_file_path, index=True, header=True
                    )
                    print(f"saving results to {eval_file_path}")

                    eval_detailed_results_df_by_instruction_dict[
                        instruction
                    ] = eval_detailed_results_df

                    instruction_score_dict[instruction] = eval_average_score

                    eval_log[f"{i_epoch}_{i_step}_{eval_idx + eval_cnt}"] = (
                        instruction,
                        eval_average_score,
                        i_epoch,
                        i_step,
                    )

                current_instruction2score = sorted(
                    current_instruction2score.items(), key=lambda x: x[1], reverse=True
                )
                print("currect_instruction2score:\n", current_instruction2score)                

                # ================== update instruction ==================
                prev_best_instruction = list(set(eval_step2max_instruction.values()))
                
                update_instruction = ""
                for i in range(len(current_instruction2score)):
                    current_instruction = current_instruction2score[i][0]
                    current_score = current_instruction2score[i][1]
                    if current_instruction not in prev_best_instruction:
                        update_instruction = current_instruction
                        eval_step2max_score[i_epoch * steps + i_step] = current_score
                        eval_step2max_instruction[i_epoch * steps + i_step] = current_instruction
                        break
                    
                if len(update_instruction) == 0:
                    update_instruction = current_instruction2score[0][0]
                    eval_step2max_score[i_epoch * steps + i_step] = current_instruction2score[0][1]
                    eval_step2max_instruction[i_epoch * steps + i_step] = current_instruction2score[0][0]
                    
                old_instructions_and_scores_eval.append(
                    (eval_step2max_instruction[i_epoch * steps + i_step], eval_step2max_score[i_epoch * steps + i_step], i_epoch, i_step)
                )
                    
                if gradient_name in {"feedback"}:
                    improved_score = (
                        eval_step2max_score[i_epoch * steps + i_step]
                        - eval_step2max_score[i_epoch * steps + i_step - 1]
                    )
                    assert meta_gradient_texts[-1][1] == -100
                    meta_gradient_texts[-1][1] = improved_score
                    
                print(
                    f"===================== update instruction: {update_instruction} ====================="
                )

                test_detailed_results_df = eval_utils.evaluate_single_instruction(
                    scorer_llm_name=scorer_llm_name,
                    instruction=update_instruction,
                    data=test_data,
                    eval_index_all=test_index,
                    tokenizer=tokenizer,
                    call_server_func=call_scorer_server_func,
                    dataset_name=dataset_name,
                    task_name=task,
                    extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
                    include_qa=include_qa,
                    instruction_pos=instruction_pos,
                    is_multiple_choice=is_multiple_choice,
                    evaluate_generated_ins_on_few_shot=evaluate_generated_ins_on_few_shot,
                    few_shot_data=few_shot_data,
                    few_shot_num=few_shot_num,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_treat_as_rouge=prediction_treat_as_rouge,
                    prediction_num_decimals=0,
                )

                test_scores = test_detailed_results_df["accuracy"]
                test_average_score = np.average(test_scores)
                print(
                    f"Epoch {i_epoch}, Step {i_step}, instruction: {update_instruction}, score: {test_average_score}"
                )

                test_epoch_step_folder = os.path.join(
                    test_result_by_instruction_folder, str(i_epoch), str(i_step)
                )
                os.makedirs(test_epoch_step_folder, exist_ok=True)
                test_filename = eval_utils.instruction_to_filename(update_instruction)
                test_file_path = os.path.join(
                    test_epoch_step_folder, f"""{test_filename}.csv"""
                )
                test_detailed_results_df.to_csv(test_file_path, index=True, header=True)
                print(f"saving results to {test_file_path}")

                test_detailed_results_df_by_instruction_dict[
                    update_instruction
                ] = test_detailed_results_df

                test_log[f"{i_epoch}_{i_step}"] = (
                    update_instruction,
                    test_average_score,
                    i_epoch,
                    i_step,
                )

                # ===================== save up-to-date results ===========================
                results_dict = dict()
                results_dict["meta_gradient_prompts"] = meta_gradient_prompts
                results_dict["meta_gradient_texts"] = meta_gradient_texts
                results_dict["selected_gradient_history_list"] = selected_gradient_history_list
                results_dict["meta_fo_momentum_gradient_prompts"] = meta_fo_momentum_gradient_prompts
                results_dict["meta_fo_momentum_gradient_texts"] = meta_fo_momentum_gradient_texts
                results_dict["selected_para_history_list"] = selected_para_history_list
                results_dict["meta_fo_momentum_para_prompts"] = meta_fo_momentum_para_prompts
                results_dict["meta_fo_momentum_para_texts"] = meta_fo_momentum_para_texts
                results_dict["modify_word_num_history"] = modify_word_num_history
                results_dict["meta_prompts"] = meta_prompts
                results_dict["meta_generate_instructions"] = meta_generate_instructions
                results_dict["setting_dict"] = setting_dict_for_save
                results_dict["train_log"] = train_log
                results_dict["eval_log"] = eval_log
                results_dict["test_log"] = test_log

                # add result eval test
                with open(os.path.join(result_by_instruction_folder, task, "results_dict.json"), "w") as fp:
                    json.dump(results_dict, fp)
                print(f"\nsaved all results to\n{save_folder}")
