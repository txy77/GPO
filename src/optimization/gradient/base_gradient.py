import os
import sys

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.optimization.gradient.feedback_gradient import Feedback_Gradient

name2gradient_method = {
    "feedback": Feedback_Gradient,
}

class Base_Gradient():
    def __init__(self, gradient_name, *args, **kwargs) -> None:
        gradient_class = name2gradient_method[gradient_name]
        self.gradient_class = gradient_class(*args, **kwargs)
    
    def gen_gradient_text(self, *args, **kwargs):
        return self.gradient_class.gen_gradient_text(*args, **kwargs)