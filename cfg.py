import numpy as np
import random
import torch
from transformers import AutoTokenizer
import json
import os
import sys

import pandas as pd
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader

import time
import numpy as np


class Config:
    workdir=os.path.dirname(os.path.abspath(__file__)) #TODO change workdir

    num_proc=10
    seed=42 #随机种子
    model_name="./deberta-v3-large" # TODO:microsoft/deberta-v3-large
    modelsavepath = workdir+'/model_weight/'
    data_path= "./dataset/"

    # split_by_paragraph=True #False to split by sentence windows
    max_length=1024 # for sentence sliding windows
    stride=128 # overlap count


    batch_size = 8 # TODO:change batch size
    logging_steps=100
    epochs=5
    lr=1e-5
    weight_decay=0.02
    accumulation_steps=1
    evaltimes=3
    num_warmup_steps=0

    resume_train_epoch=0


#设置随机种子,保证模型可以复现
def seed_everything():
    seed=Config.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()
