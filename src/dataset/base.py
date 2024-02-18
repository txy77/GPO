import os
import sys

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.dataset.bbh import BBH_Dataset
from src.dataset.mmlu import MMLU_Dataset
from src.dataset.webnlg import WebNLG_Dataset
from src.dataset.wsc import WSC_Dataset
from src.dataset.gsm8k import GSM8K_Dataset

name2dataset = {
    "bbh": BBH_Dataset,
    "mmlu": MMLU_Dataset,
    "webnlg": WebNLG_Dataset,
    "wsc": WSC_Dataset,
    "gsm8k": GSM8K_Dataset,
}

class Base_Dataset():
    def __init__(self, dataset_name, *args, **kwargs) -> None:
        dataset_class = name2dataset[dataset_name]
        self.dataset_class = dataset_class(*args, **kwargs)
        
    def get_ratio(self):
        return self.dataset_class.get_ratio()
    
    def read_data(self, *args, **kwargs):
        return self.dataset_class.read_data(*args, **kwargs)
    
    def get_single_question(self, data, idx):
        return self.dataset_class.get_single_question(data, idx)
    
    def get_single_answer(self, data, idx):
        return self.dataset_class.get_single_answer(data, idx)
    
    def get_single_solution(self, data, idx):
        return self.dataset_class.get_single_solution(data, idx)
    
    def get_task_setting(self, *args, **kwargs):
        return self.dataset_class.get_task_setting(*args, **kwargs)
    
    def get_q(self, *args, **kwargs):
        return self.dataset_class.get_q(*args, **kwargs)
    
    def get_a(self, *args, **kwargs):
        return self.dataset_class.get_a(*args, **kwargs)