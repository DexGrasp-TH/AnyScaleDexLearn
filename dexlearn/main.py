import os
import sys
import random
import logging
import traceback

import hydra
from omegaconf import DictConfig
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dexlearn.utils.config import resolve_type_supervision_config
from task import *


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    resolve_type_supervision_config(cfg)
    eval(f"task_{cfg.task_name}")(cfg)


if __name__ == "__main__":
    main()
