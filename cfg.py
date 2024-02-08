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


import numpy as np



class Config:
    workdir=os.path.dirname(os.path.abspath(__file__))

    seed=2024#随机种子
    model_name="microsoft/deberta-v3-large"
    savepath = workdir+'//modelsave'
    data_path="dataset/"

    max_sen_count=15
    stride=3 # overlap count

    batch_size=5 # TODO:change batch size
    epochs=5
    lr=2e-5
    weight_decay=0.01
    accumulation_steps=1 # batch size不同，不能直接除。有空再修
    evaltimes=5
    logging_steps=50
    num_warmup_steps=0

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


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["berttokenids"] for sample in batch]
        output["type_ids"] = [sample["berttokentoken_type_ids"] for sample in batch]
        output["mask"] = [sample["berttokenmask"] for sample in batch]
        output["targets"] = [sample["bertlabels"] for sample in batch]
        
         # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
            output["type_ids"] = [s + (batch_max - len(s)) * [0] for s in output["type_ids"]]
            output["targets"] = [s + (batch_max - len(s)) * [-100] for s in output["targets"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]
            output["type_ids"] = [(batch_max - len(s)) * [0] + s for s in output["type_ids"]]
            output["targets"] = [(batch_max - len(s)) * [-100] + s for s in output["targets"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["type_ids"] = torch.tensor(output["type_ids"], dtype=torch.long)
        output["targets"] = torch.tensor(output["targets"], dtype=torch.long)

        return output
