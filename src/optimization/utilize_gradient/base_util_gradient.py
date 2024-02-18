import os
import sys

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.optimization.utilize_gradient.generate_based_util_gradient import Generate_based_util_gradient
from src.optimization.utilize_gradient.edit_based_util_gradient import Edit_based_util_gradient
from src.optimization.utilize_gradient.generate_without_gradient import Generate_Without_Gradient
from src.optimization.utilize_gradient.edit_without_gradient import Edit_Without_Gradient


name2util_gradient_method = {
    "generate": Generate_based_util_gradient,
    "edit": Edit_based_util_gradient,
    "generate_without": Generate_Without_Gradient,
    "edit_without": Edit_Without_Gradient,
}

class Base_Util_Gradient():
    def __init__(self, util_gradient_name, *args, **kwargs) -> None:
        util_gradient_class = name2util_gradient_method[util_gradient_name]
        self.util_gradient_class = util_gradient_class(*args, **kwargs)
    
    def gen_util_gradient_text(self, *args, **kwargs):
        return self.util_gradient_class.gen_util_gradient_text(*args, **kwargs)