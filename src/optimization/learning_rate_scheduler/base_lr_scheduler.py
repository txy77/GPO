import os
import sys

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.optimization.learning_rate_scheduler.fixed import Fixed_LR_Scheduler
from src.optimization.learning_rate_scheduler.linear import Linear_LR_Scheduler
from src.optimization.learning_rate_scheduler.consine import Consine_LR_Scheduler

name2step_size_strategy = {
    "fixed": Fixed_LR_Scheduler,
    "linear": Linear_LR_Scheduler,
    "consine": Consine_LR_Scheduler
}

class Base_LR_Scheduler():
    def __init__(self, decay_strategy, *args, **kwargs) -> None:
        step_size_strategy_class = name2step_size_strategy[decay_strategy]
        self.step_size_strategy_class = step_size_strategy_class(*args, **kwargs)
    
    def calculate_step_size(self, *args, **kwargs):
        return self.step_size_strategy_class.calculate_step_size(*args, **kwargs)