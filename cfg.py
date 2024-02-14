import numpy as np#进行矩阵运算的库
import random#提供了一些用于生成随机数的函数
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

    seed=42 #随机种子
    model_name="microsoft/deberta-v3-base" # TODO:microsoft/deberta-v3-large
    modelsavepath = workdir+'//modelsave/'
    data_path=workdir+"/dataset/"

    split_by_paragraph=True #False to split by sentence windows
    max_sen_count=8 # for sentence sliding windows 
    stride=3 # overlap count

    # O judge threshold
    threshold=0.9

    batch_size=10 # TODO:change batch size
    logging_steps=100
    epochs=3
    lr=2e-5
    weight_decay=0.01
    accumulation_steps=1 # batch size不同，不能直接除。有空再修
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
