import os
import sys

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.optimization.first_order_momentum.selection_methods.recency_selection import Recency_Selection
from src.optimization.first_order_momentum.selection_methods.relavance_selection import Relavance_Selection
from src.optimization.first_order_momentum.selection_methods.importance_selection import Importance_Selection
from src.optimization.first_order_momentum.momentum_update_methods.real_time_update import Real_time_Update
from src.optimization.first_order_momentum.momentum_update_methods.k_update import K_Update

selection_method2class = {
    "recency": Recency_Selection,
    "relavance": Relavance_Selection,
    "importance": Importance_Selection,
}

momentum_update_method2class = {
    "real-time": Real_time_Update,
    "k-list": K_Update
}

class Base_FO_Momemtum():
    def __init__(self, selection_method_name, momentum_update_method_name) -> None:
        if momentum_update_method_name == "k-list":
            selection_method_class = selection_method2class[selection_method_name]
            self.selection_method_class = selection_method_class()
        momentum_update_method_class = momentum_update_method2class[momentum_update_method_name]
        self.momentum_update_method_class = momentum_update_method_class()
    
    def select(self, *args, **kwargs):
        return self.selection_method_class.select(*args, **kwargs)
    
    def update_momentum(self, *args, **kwargs):
        return self.momentum_update_method_class.update_momentum(*args, **kwargs)