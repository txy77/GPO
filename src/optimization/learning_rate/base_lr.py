import os
import sys

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.optimization.learning_rate.wo_lr import Wo_LR
from src.optimization.learning_rate.w_lr import W_LR

name2lr_method = {
    "wo_lr": Wo_LR,
    "w_lr": W_LR,
}

class Base_LR():
    def __init__(self, lr_name, *args, **kwargs) -> None:
        lr_class = name2lr_method[lr_name]
        self.lr_class = lr_class(*args, **kwargs)
    
    def use_learning_rate(self, *args, **kwargs):
        return self.lr_class.use_learning_rate(*args, **kwargs)