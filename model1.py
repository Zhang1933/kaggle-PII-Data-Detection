import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoModel
from cfg import *

class NLPModel(nn.Module):
    def __init__(self,id2label,label2id,modelname,training=True):
        super().__init__()
    #     self.model=AutoModelForTokenClassification.from_pretrained(modelname, num_labels=len(id2label),
    # id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)#label映射到id
        self.model = AutoModel.from_pretrained(modelname, # num_labels=len(id2label),
    id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)
        self.lossfunc = nn.CrossEntropyLoss()
        self.num_labels=len(id2label)
        # if training: self.model.save_pretrained(Config.modelsavepath)

        hidden_dim = 1024
        self.lstm = nn.LSTM(input_size = hidden_dim,
                            hidden_size = hidden_dim  // 2,
                            num_layers = 1,
                            batch_first=True,
                            bidirectional=True,
                            # dropout=0.1
                        )
        self.l = nn.Linear(hidden_dim, self.num_labels)

    def forward(self,input_ids, token_type_ids,attention_mask,labels=None):
        output = self.model(input_ids, token_type_ids, attention_mask)
        # batchsize*seq length
        # logit=output[0]

        self.lstm.flatten_parameters()
        hidden, (_, _) = self.lstm(output[0])
        logit = self.l(hidden)
        loss=0
        logit=logit.view(-1,logit.shape[-1])
        if labels is not None:
             labels=labels.view(-1)
             loss=self.lossfunc(logit,labels)
        
        
        logit=logit.detach().cpu().numpy()
        return logit,loss
