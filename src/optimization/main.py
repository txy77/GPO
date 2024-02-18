import os
import sys
import datetime
import time
from absl import app
from absl import flags

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.optimization.args import get_args
from src.optimization.optimize import prompt_optimization

def main(_):
    optimization_kwargs = get_args()
    prompt_optimization(**optimization_kwargs)
    print("FINISHED!")

if __name__ == '__main__':
    app.run(main)