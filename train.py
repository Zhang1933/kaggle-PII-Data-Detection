from cfg import *
from util import *
Config.modelsavepath
from itertools import chain
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, ConcatDataset
from transformers import DataCollatorForTokenClassification
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import gc
from model1 import NLPModel


##### main
data = json.load(open("./dataset/train.json"))

external = json.load(open("./combine_data/mixtral-8x7b-v1.json"))
# print("external datapoints: ", len(external))
external2 = json.load(open("./combine_data/mpware_mixtral8x7b_v1.1-no-i-username.json"))
# external2 = json.load(open("./combine_data/mpware_mixtral_clean.json"))
# print("external2 datapoints: ", len(external2))
p = []
for i in external2:
    if any(np.array(i["labels"]) != "O"):
        p.append(i)
# moredata = json.load(open(Config.data_path+"/moredata_dataset_fixed.json"))
# print("moredata datapoints: ", len(moredata))

data = data+external+p
# print("combined: ", len(data))
data_pd=pd.DataFrame(data)

tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

# 将所有数据的labels连接在一起,然后查重,转成list的格式,然后从小到大排序
all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
print(f"all_labels:{all_labels}")
label2id = {l: i for i,l in enumerate(all_labels)}
#这个是{id:label},上面是{label:id}
id2label = {v:k for k,v in label2id.items()}

# Open a file for writing
with open(Config.modelsavepath+"//idlabel.json", "w") as f:
    # Write the map to the file in JSON format
    json.dump({'label2id':label2id,'id2label':id2label}, f)


ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [str(x["document"]) for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "labels": [x["labels"] for x in data]
})

preprocesssed_ds=ds.map(train_preprocesss, fn_kwargs={'tokenizer':tokenizer,'label2id':label2id},num_proc=Config.num_proc,desc="prepocessing data")

# rebuid dataset 
tmp_pd=expanddataset(preprocesssed_ds)
# print(tmp_pd['berttokenids'].str.len().agg(['mean','max','std','min']))
full_ds=Dataset.from_pandas(tmp_pd)

# ============================== vaild data generate ================================
vaild_data = json.load(open("./combine_data/pii_dataset_fixed.json"))
vaild_data = vaild_data[:len(vaild_data) * 7 // 10]
vaild_ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in vaild_data],
    "document": [str(x["document"]) for x in vaild_data],
    "tokens": [x["tokens"] for x in vaild_data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in vaild_data],
    "labels": [x["labels"] for x in vaild_data]
})
vaild_preprocesssed_ds=vaild_ds.map(train_preprocesss, fn_kwargs={'tokenizer':tokenizer,'label2id':label2id},num_proc=Config.num_proc,desc="prepocessing data")
# rebuid dataset 
tmp_pd=expanddataset(vaild_preprocesssed_ds)
vaild_full_ds=Dataset.from_pandas(tmp_pd)
print(f"vaild_full_ds:{len(vaild_full_ds)}")
# =====================================================================================
# train_add_ds = Dataset.from_dict({
#     "full_text": [x["full_text"] for x in train_data_add],
#     "document": [str(x["document"]) for x in train_data_add],
#     "tokens": [x["tokens"] for x in train_data_add],
#     "trailing_whitespace": [x["trailing_whitespace"] for x in train_data_add],
#     "labels": [x["labels"] for x in train_data_add]
# })
# train_add_preprocesssed_ds=train_add_ds.map(train_preprocesss, fn_kwargs={'tokenizer':tokenizer,'label2id':label2id},num_proc=Config.num_proc,desc="prepocessing data")
# # rebuid dataset 
# tmp_pd=expanddataset(train_add_preprocesssed_ds)
# train_add_ds=Dataset.from_pandas(tmp_pd)
# print(f"train_add_ds:{len(train_add_ds)}")
# ===================================================================================
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# train_len=int(0.90*len(full_ds)) # 长度分割，兼容老版本torch
# train_dataset, val_dataset = random_split(full_ds, [train_len,len(full_ds)-train_len],generator=torch.Generator().manual_seed(Config.seed)) #TODO:改变比例

train_dataset, val_dataset = full_ds, vaild_full_ds 
print(f"train dataset len:{len(train_dataset)}")
print(f"vail dataset len:{len(val_dataset)}")

data_collator = Collate(tokenizer=tokenizer)
train_dataloader=DataLoader(train_dataset,batch_size=Config.batch_size,pin_memory=True,collate_fn=data_collator,shuffle=True)
val_dataloader=DataLoader(ConcatDataset([val_dataset, vaild_full_ds]),batch_size=Config.batch_size * 2,pin_memory=True,collate_fn=data_collator)


model=NLPModel(id2label,label2id,Config.model_name).to(device)
# # 分层试一下,能不能保留泛化性能
# optimizer= AdamW([{"params": model.model.parameters(),"lr":2e-6},
#                   {"params": model.lstm.parameters(),"lr":1e-5},
#                   {"params": model.l.parameters(),"lr":1e-5}
#                 ],lr=Config.lr,weight_decay=Config.weight_decay)
optimizer= AdamW(model.parameters(),lr=Config.lr,weight_decay=Config.weight_decay)
scaler = torch.cuda.amp.GradScaler()
valstep = len(train_dataloader) // Config.evaltimes

num_train_steps = int(len(train_dataloader) * Config.epochs)
scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=Config.num_warmup_steps, num_training_steps=num_train_steps, 
            )

def compute_metrics(predictions,traget, all_labels):
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, traget)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, traget)
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

def validation(val_dataloader,model):
    model.eval()

    allprecitions=[]
    alltargets=[]
    for step,dataset in enumerate(tqdm(val_dataloader)):

        with torch.no_grad():
            ids = dataset["ids"].to(device,non_blocking=True)
            mask = dataset["mask"].to(device,non_blocking=True)
            targets = dataset["targets"].to(device,non_blocking=True)
            tokentype = dataset["type_ids"].to(device,non_blocking=True)
            logit,loss = model(ids,mask, tokentype,targets)

            targets=targets.view(-1)
            targets = targets.detach().cpu().numpy()

            allprecitions.append(np.argmax(logit, axis=1))
            alltargets.append(targets)
        del ids,mask,targets,tokentype,loss,logit

    score=compute_metrics(allprecitions,alltargets,all_labels)
    
    gc.collect()
    torch.cuda.empty_cache()

    model.train()

    return score

tokenizer.save_pretrained(Config.modelsavepath)

gc.collect()
torch.cuda.empty_cache()
start_time=time.time()
bestscore=0
epoch=0

if Config.resume_train_epoch:
    checkpoint = torch.load(Config.workdir+f"/checkpoint/checkpoint{Config.resume_train_epoch}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    bestscore=checkpoint['bestscore']
    model.train()
    print(f"start epoch:{epoch}")

while epoch <Config.epochs:
    losses=[] 
    for step,dataset in enumerate(tqdm(train_dataloader)):
        with torch.cuda.amp.autocast():
            ids = dataset["ids"].to(device,non_blocking=True)
            mask = dataset["mask"].to(device,non_blocking=True)
            tokentype = dataset["type_ids"].to(device,non_blocking=True)
            targets = dataset["targets"].to(device,non_blocking=True)

            logit,loss = model(ids,mask, tokentype,targets)
            loss/= Config.accumulation_steps

        # Accumulates scaled gradients.
        scaler.scale(loss).backward()
        # losses.append(loss.item())
        
        if (step+1) % Config.accumulation_steps == 0:
            scaler.step(optimizer)
            scale = scaler.get_scale()
            # Updates the scale for next iteration.
            scaler.update()
            skip_lr_sched = (scale != scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step()
            optimizer.zero_grad()

        # validation
        if epoch > 0 and (step+1) % valstep == 0:
            scores=validation(val_dataloader,model)
            print("\nValidation:Epoch{} Step [{}/{}] Loss: {} Time: {:.1f} Metric: {}".format(epoch,step, len(train_dataloader)-1, loss.item(), time.time()-start_time,scores))
            if scores['f1']>bestscore:
                print(f"Best score is {bestscore} → {scores['f1']}. Saving model")
                bestscore=scores['f1']
                torch.save(model.state_dict(), os.path.join(Config.modelsavepath,"seed{}_e{}_vs{}_score{:.3f}.pth".format(Config.seed,epoch,step+1,bestscore)))
            else:
                print("\nno improvement, bestf1 is {:.3f} score is {}".format(bestscore,scores))
                torch.save(model.state_dict(), os.path.join(Config.modelsavepath,"seed{}_e{}_vs{}_score{:.3f}.pth".format(Config.seed,epoch,step+1,scores['f1'])))
                # report loss
        if (step+1) % Config.logging_steps == 0:
            # print("Logging: Step [{}/{}] Loss: {:.3f} Time: {:.1f}".format(step, len(train_dataloader)-1, loss.item(), time.time()-start_time))
            gc.collect()
            torch.cuda.empty_cache()
        del ids,mask,targets,tokentype,loss,logit

    epoch+=1
    Config.resume_train_epoch += 1
    # checkpoint
    torch.save({
            'epoch': epoch,
            'bestscore':bestscore,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'scaler_state_dict':scaler.state_dict()
            }, Config.workdir+"/checkpoint/checkpoint{}.pth".format(epoch))

tokenizer.save_pretrained(Config.modelsavepath)