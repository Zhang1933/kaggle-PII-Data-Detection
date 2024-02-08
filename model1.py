import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from cfg import *

class NLPModel(nn.Module):
    def __init__(self,id2label,label2id):
        super().__init__()
        self.model=AutoModelForTokenClassification.from_pretrained(Config.model_name, num_labels=len(id2label),
    id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)#label映射到id
        self.lossfunc = nn.CrossEntropyLoss()
        self.num_labels=len(id2label)

    def forward(self,input_ids, token_type_ids,attention_mask,labels=None):
        output = self.model(input_ids, token_type_ids,attention_mask)
        # batchsize*seq length*12
        logit=output[0]
        loss=0
        logit=logit.view(-1,logit.shape[-1])
        if labels is not None:
             labels=labels.view(-1)
             loss=self.lossfunc(logit,labels)
        
        
        logit=logit.detach().cpu().numpy()
        return logit,loss
