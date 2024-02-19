"""The aruguments of prompt optimization."""

import os
import sys
import datetime
from absl import flags

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, GPO_ROOT_PATH)
ROOT_DATA_FOLDER_PATH = os.path.join(GPO_ROOT_PATH, "data")


# basic config
_OPENAI_API_KEY = flags.DEFINE_string("openai_api_key", "", "The OpenAI API key.")

_OPENAI_API_KEY_LIST = flags.DEFINE_list(
    "openai_api_key_list", "", "The OpenAI API key list."
)

_GPUS = flags.DEFINE_list(
    "gpus",
    ["0"],
    "The list of GPUs to use. If multiple GPUs are provided, the model will be used",
)

# dataset config
_DATASET = flags.DEFINE_string(
    "dataset", "gsm8k", "The name of dataset to search for instructions on."
)

# task config
_TASK = flags.DEFINE_list(
    "task_name", "all", "The name of the task to search for instructions on."
)

# optimizer config
_OPTIMIZER = flags.DEFINE_string(
    "optimizer_llm_name", "gpt-3.5-turbo", "The name of the optimizer LLM."
)

_OPTIMIZER_TEMPERATURE = flags.DEFINE_float(
    "optimizer_temperature", 1.0, "temperature of optimizer"
)

_INITIAL_INSTRUCTION = flags.DEFINE_string(
    "initial_instruction",
    "Let's think step by step.",
    "The initial instructions to search for.",
)

_NUM_SEARCH_EPOCHS = flags.DEFINE_integer(
    "num_search_epochs",
    3,
    "The number of search steps to optimize prompt",
)

_INSTRUCTION_POS = flags.DEFINE_string(
    "instruction_pos",
    "Q_end",
    "The position of the instruction to search for.",
)

_NUM_GENERATED_INSTRUCTIONS_IN_EACH_STEP = flags.DEFINE_integer(
    "num_generated_instructions_in_each_step",
    8,
    "The number of instructions to generate in each step.",
)

_GRADIENT_NAME = flags.DEFINE_string(
    "gradient_name",
    "feedback",
    "The name of the gradient to use for the optimization.",
)

_MOMENTUM_PARA_NAME = flags.DEFINE_string(
    "momentum_para_name",
    "-",
    "The name of the momentum para to use for the optimization.",
)

_MOMENTUM_SELECTION_NAME = flags.DEFINE_string(
    "momentum_selection_name",
    "-",
    "The name of the momentum selection to use for the optimization.",
)

_MOMENTUM_SELECTION_NUM = flags.DEFINE_integer(
    "momentum_selection_num",
    0,
    "Number of momentum selection to use for the optimization.",
)

_MOMENTUM_UPDATE_NAME = flags.DEFINE_string(
    "momentum_update_name",
    "-",
    "The name of the momentum update to use for the optimization.",
)

# learning_rate_config

_LEARNING_RATE_NAME = flags.DEFINE_string(
    "learning_rate_name",
    "wo_lr",
    "The name of the learning rate use for optimization"
)

_INITIAL_STEP_SIZE = flags.DEFINE_integer(
    "initial_step_size",
    0,
    "The number of the step size use for optimization"
)

_DECAY_STRATEGY = flags.DEFINE_string(
    "decay_strategy",
    "-", # {'-', 'fixed', 'linear', 'consine'}
    "The type of decay strategy to use"
)

_USE_WARMUP_STRATEGY = flags.DEFINE_bool(
    "use_warmup_strategy",
    False, 
    "Whether to use warmup strategy"
)

_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps",
    5,
    "The number of steps to warmup"
)

_FINAL_STEP_SIZE = flags.DEFINE_integer(
    "final_step_size",
    0,
    "The final step size"
)

_UTIL_GRADIENT_NAME = flags.DEFINE_string(
    "util_gradient_name",
    "-",
    "The name of the utilize gradient to use for the optimization.",
)

_OPT_BATCH_SIZE = flags.DEFINE_integer(
    "opt_batch_size",
    8,
    "The batch size for the optimization.",
)

_FORMAT_DATA_NUM = flags.DEFINE_integer(
    "format_data_num",
    3,
    "The number of data to format for utilize gradient to optimize the prompt.",
)


# evaluation config

_SCORER = flags.DEFINE_string(
    "scorer_llm_name", "llama2-chat-7b", "The name of the scorer LLM."
)

_INCLUDE_QA = flags.DEFINE_bool(
    "include_qa", False, "Whether to include QA pairs in the prompt."
)

_ONLY_EVALUATE = flags.DEFINE_bool(
    "only_evaluate", False, "Whether to only evaluate and not optimize"
)

_EVALUATE_FEW_SHOT = flags.DEFINE_bool(
    "evaluate_generated_ins_on_few_shot", False, "Whether to evaluate on few shot"
)

_FEW_SHOT_NUM = flags.DEFINE_integer(
    "few_shot_number", 5, "The number of sample in few shot"
)


def get_args():
    # basic config
    openai_api_key = _OPENAI_API_KEY.value
    openai_api_key_list = _OPENAI_API_KEY_LIST.value
    gpus = _GPUS.value
    
    # dataset config
    dataset_name = _DATASET.value.lower()
    task_name = _TASK.value
    
    # optimization config
    optimizer_llm_name = _OPTIMIZER.value
    optimizer_temperature = _OPTIMIZER_TEMPERATURE.value
    initial_instruction = _INITIAL_INSTRUCTION.value
    instruction_pos = _INSTRUCTION_POS.value
    opt_batch_size = _OPT_BATCH_SIZE.value
    format_data_num = _FORMAT_DATA_NUM.value
    gradient_name = _GRADIENT_NAME.value
    
    # first order optimization
    momentum_para_name = _MOMENTUM_PARA_NAME.value
    momentum_selection_name = _MOMENTUM_SELECTION_NAME.value
    momentum_selection_num = _MOMENTUM_SELECTION_NUM.value
    momentum_update_name = _MOMENTUM_UPDATE_NAME.value
    
    # learning rate
    learning_rate_name = _LEARNING_RATE_NAME.value
    initial_step_size = _INITIAL_STEP_SIZE.value
    decay_strategy = _DECAY_STRATEGY.value
    use_warmup_strategy = _USE_WARMUP_STRATEGY.value
    warmup_steps = _WARMUP_STEPS.value
    final_step_size = _FINAL_STEP_SIZE.value
    
    util_gradient_name = _UTIL_GRADIENT_NAME.value
    
    num_search_epochs = _NUM_SEARCH_EPOCHS.value
    num_generated_instructions_in_each_step = (
        _NUM_GENERATED_INSTRUCTIONS_IN_EACH_STEP.value
    )
    
    # evaluation config
    scorer_llm_name = _SCORER.value
    include_qa = _INCLUDE_QA.value
    evaluate_generated_ins_on_few_shot = _EVALUATE_FEW_SHOT.value  # 0-shot or few-shot
    few_shot_num = _FEW_SHOT_NUM.value
    only_evaluate = _ONLY_EVALUATE.value
    
    # save config
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    save_folder = os.path.join(
        GPO_ROOT_PATH,
        "outputs",
        "optimization-results",
        f"{dataset_name.upper()}-s-{scorer_llm_name}-o-{optimizer_llm_name}-{datetime_str}/",
    )

    result_by_instruction_folder = os.path.join(save_folder, "result_by_instruction")
    os.makedirs(result_by_instruction_folder, exist_ok=True)
    print(f"result directory:\n{save_folder}")

    assert instruction_pos in {
        "before_Q",
        "Q_begin",
        "Q_end",
        "A_begin",
    }
    
    assert gradient_name in {
        "feedback",
        "-"
    }
    
    assert momentum_para_name in {
        "-",
        "para",
        "feedback",
    }
    
    assert momentum_selection_name in {
        "-",
        "recency",
        "relavance",
        "importance",
    }
    
    assert momentum_update_name in {
        "-",
        "k-list",
        "real-time"
    }
    
    assert learning_rate_name in {
        "wo_lr",
        "w_lr"
    }
    
    assert decay_strategy in {
        "-",
        "fixed",
        "linear",
        "consine",
    }
    
    assert util_gradient_name in {
        "edit",
        "generate",
        "generate_without",
        "edit_without",
    }

    optimization_kwargs = {
        # basic config
        "openai_main_api_key": openai_api_key,
        "openai_api_key_list": openai_api_key_list,
        "gpus": gpus,
        
        # dataset config
        "dataset_name": dataset_name,
        "task_name": task_name,
        
        # optimization config
        "optimizer_llm_name": optimizer_llm_name,
        "optimizer_temperature": optimizer_temperature,
        "initial_instruction": initial_instruction,
        "instruction_pos": instruction_pos,
        "opt_batch_size": opt_batch_size,
        "format_data_num": format_data_num,
        "gradient_name": gradient_name,
        
        # first order momentum
        "momentum_para_name": momentum_para_name,
        "momentum_selection_name": momentum_selection_name,
        "momentum_selection_num": momentum_selection_num,
        "momentum_update_name": momentum_update_name,
        
        # learning rate
        "learning_rate_name": learning_rate_name,
        "initial_step_size": initial_step_size,
        "decay_strategy": decay_strategy,
        "use_warmup_strategy": use_warmup_strategy,
        "warmup_steps": warmup_steps,
        "final_step_size": final_step_size,
        
        # util gradient
        "util_gradient_name": util_gradient_name,
        
        "num_search_epochs": num_search_epochs,
        "num_generated_instructions_in_each_step": (
            num_generated_instructions_in_each_step
        ),
        
        # evaluation config
        "scorer_llm_name": scorer_llm_name,
        "include_qa": include_qa,
        "evaluate_generated_ins_on_few_shot": evaluate_generated_ins_on_few_shot,
        "few_shot_num": few_shot_num,
        "only_evaluate": only_evaluate,
        
        # save config
        "result_by_instruction_folder": result_by_instruction_folder,
        "save_folder": save_folder,        
    }

    return optimization_kwargs
