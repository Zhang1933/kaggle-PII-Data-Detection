#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import argparse
from itertools import chain
from functools import partial

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate
from datasets import Dataset, features
import numpy as np


# In[2]:


TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 1280
OUTPUT_DIR = 'output'


# In[3]:


import json

# data = json.load(open("kaggle/input/pii-detection-removal-from-educational-data/train.json"))
# # data2 = json.load(open("kaggle/input/pii-detection-removal-from-educational-data/mixtral-8x7b-v1.json"))

# # for i in range(len(data2)):
# #     data2[i]["document"] = len(data) + i

# data.extend(data2)

# print(len(data))
# print(data[0].keys())


data = json.load(open("kaggle/input/pii-detection-removal-from-educational-data/train.json"))

# # downsampling of negative examples
p=[] # positive samples (contain relevant labels)
n=[] # negative samples (presumably contain entities that are possibly wrongly classified as entity)
for d in data:
    if any(np.array(d["labels"]) != "O"): p.append(d)
    else: n.append(d)
print("original datapoints: ", len(data))

external = json.load(open("kaggle/input/pii-detection-removal-from-educational-data/pii_dataset_fixed.json"))
print("external datapoints: ", len(external))

for i in range(len(external)):
    external[i]["document"] = len(data) + i
    for j in range(len(external[i]["labels"])):
        external[i]["labels"][j] = str(external[i]["labels"][j])


moredata = json.load(open("kaggle/input/pii-detection-removal-from-educational-data/moredata_dataset_fixed.json"))
print("moredata datapoints: ", len(moredata))

for i in range(len(moredata)):
    moredata[i]["document"] = len(data) + len(external) + i
    for j in range(len(moredata[i]["labels"])):
        moredata[i]["labels"][j] = str(moredata[i]["labels"][j])

data = moredata+external+p+n[:len(n)//3]

# print("combined: ", len(data))


data2 = json.load(open("kaggle/input/pii-detection-removal-from-educational-data/mixtral-8x7b-v1.json"))

for i in range(len(data2)):
    data2[i]["document"] = len(data) + i

data.extend(data2)

print(len(data))
print(data[0].keys())


# In[4]:


all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

print(id2label)


# In[5]:


target = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
]


# In[6]:


def tokenize(example, tokenizer, label2id):
    text = []

    # these are at the character level
    labels = []
    targets = []

    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):

        text.append(t)
        labels.extend([l]*len(t))
        
        if l in target:
            targets.append(1)
        else:
            targets.append(0)
        # if there is trailing whitespace
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=True, max_length=TRAINING_MAX_LENGTH)
    
    target_num = sum(targets)
    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:

        # CLS token
        if start_idx == 0 and end_idx == 0: 
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "target_num": target_num,
        "group": 1 if target_num>0 else 0
    }


# In[7]:


tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [x["document"] for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
})


# In[ ]:


ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}, num_proc=2)
ds = ds.class_encode_column("group")




from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5*5) * recall * precision / (5*5*precision + recall) 
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results




model = AutoModelForTokenClassification.from_pretrained(
    TRAINING_MODEL_PATH,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)



FREEZE_EMBEDDINGS = False
FREEZE_LAYERS = 6
NOISE_TUNE = False

if FREEZE_EMBEDDINGS:
    print('Freezing embeddings.')
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False

# if NOISE_TUNE:
#     noise_lambda = 0.1
#     for name, para in model.named_parameters():
#         model.state_dict()[name][:] += (torch.rand(para.shape)-0.5) * noise_lambda * torch.std(para)
        
if FREEZE_LAYERS>0:
    print(f'Freezing {FREEZE_LAYERS} layers.')
    for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False


import torch.nn as nn
# from torchcrf import CRF

print(model)




final_ds = ds.train_test_split(test_size=0.2, seed=42) # cannot use stratify_by_column='group'


from transformers import EarlyStoppingCallback


args = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    fp16=True,
    # warmup_steps=100,
    learning_rate=5e-5,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    report_to="none",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    lr_scheduler_type='cosine',
    metric_for_best_model="f1",
    greater_is_better=True,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=final_ds["train"], 
    eval_dataset=final_ds["test"], 
    data_collator=collator, 
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 10


# def train(model, train_iter, test_iter, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         print('epoch: ', epoch + 1)
#         train_loss = 0
#         train_acc = 0
#         n = 0
#         for batch in train_iter:
#             optimizer.zero_grad()
#             input_ids = batch['input_ids']
#             attention_mask = batch['attention_mask']
#             labels = batch['labels']
#             loss = model(input_ids, attention_mask, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             n += 1
#         print('train loss: {:.4f}'.format(train_loss / n))
#         test_loss = 0
#         test_acc = 0
#         n = 0
#         for batch in test_iter:
#             input_ids = batch['input_ids']
#             attention_mask = batch['attention_mask']
#             labels = batch['labels']
#             with torch.no_grad():
#                 loss = model(input_ids, attention_mask, labels)
#                 test_loss += loss.item()
#                 n += 1
#         print('test loss: {:.4f}'.format(test_loss / n))

# train(model, final_ds["train"], final_ds["test"], optimizer, 10)

# In[ ]:


trainer.save_model(OUTPUT_DIR)
torch.cuda.empty_cache()


# ## Inference

# In[ ]:




