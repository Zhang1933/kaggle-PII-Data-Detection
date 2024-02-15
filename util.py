from cfg import *

if not Config.split_by_paragraph:
    from nltk.tokenize import sent_tokenize


from unidecode import unidecode
import copy
from tqdm import tqdm,tqdm_notebook

# batch 对齐
class Collate:
    def __init__(self, tokenizer,if_train=True):
        self.tokenizer = tokenizer
        self.if_train=if_train
    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["berttokenids"] for sample in batch]
        output["type_ids"] = [sample["berttokentoken_type_ids"] for sample in batch]
        output["mask"] = [sample["berttokenmask"] for sample in batch]
        if self.if_train: 
            output["targets"] = [sample["bertlabels"] for sample in batch]
        else :
            output["token_org_length"]=[len(ids) for ids in output["ids"]]

         # calculate max token length of this batch
        batch_max = max(output["token_org_length"])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
            output["type_ids"] = [s + (batch_max - len(s)) * [0] for s in output["type_ids"]]
            if self.if_train: output["targets"] = [s + (batch_max - len(s)) * [-100] for s in output["targets"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]
            output["type_ids"] = [(batch_max - len(s)) * [0] + s for s in output["type_ids"]]
            if self.if_train: output["targets"] = [(batch_max - len(s)) * [-100] + s for s in output["targets"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["type_ids"] = torch.tensor(output["type_ids"], dtype=torch.long)
        if self.if_train: output["targets"] = torch.tensor(output["targets"], dtype=torch.long)

        return output

def text_precessor(data):
    # 预处理
    # 文本转小写,unicode 轉換ascii
    # 转换失败转'*'
    data=copy.deepcopy(data)
    for i in tqdm(range(len(data)),desc="preprocess texe"):
        data[i]['full_text']=unidecode(data[i]['full_text'],errors='replace',replace_str='*')
        data[i]['full_text']=data[i]['full_text'].lower()
        for j in range(len(data[i]['tokens'])):
            data[i]['tokens'][j]=unidecode(data[i]['tokens'][j],errors='replace',replace_str='*')  
            data[i]['tokens'][j]=data[i]['tokens'][j].lower().strip()
    return data

def split_by_paragraph(text):
    res=[i+'\n\n' for i in  text.split('\n\n')]
    return res

# 分句子
def split_token(text):
    """
        分句子，返回列表
    """
    data_out=[]
    if Config.split_by_paragraph:
        doc=split_by_paragraph(text)
        return doc
    else:
        doc=sent_tokenize(text)
    split_sentence=doc
    length=len(split_sentence)
    idx=0
    while idx+Config.max_sen_count<length:
        split_sentence[idx+Config.max_sen_count-Config.stride]="[SEP]"+split_sentence[idx+Config.max_sen_count-Config.stride]
        if idx==0:
            data_out.append("".join(split_sentence[idx:idx+Config.max_sen_count]))
        else:
            data_out.append("".join(split_sentence[idx-Config.stride:idx+Config.max_sen_count]))
        idx+=Config.max_sen_count
    # deal with reminder
    if idx<Config.max_sen_count:# few sentences,concatenate directly
        data_out.append("".join(split_sentence[idx:length]))
    else:
        data_out.append("".join(split_sentence[idx-Config.stride:length]))
    return data_out

def train_preprocesss(example,tokenizer,label2id):
    # rebuild text from tokens

    example['bertlabels']=[]
    example['berttokenids']=[]
    example['berttokenmask']=[]
    example['berttokentoken_type_ids']=[]

    tokens_split_list=[]
    trailing_whitespace_split_list=[]
    labels_split_list=[]

    right_idx=0
    for i in range(0,len(example['tokens'])):
        if example['tokens'][i] == '\n\n':
            tokens_split_list.append(example['tokens'][right_idx:i+1])
            trailing_whitespace_split_list.append(example['trailing_whitespace'][right_idx:i+1])
            labels_split_list.append(example['labels'][right_idx:i+1])
            right_idx=i+1
    if  len(tokens_split_list)==0:
        tokens_split_list.append( example['tokens'])
        trailing_whitespace_split_list.append( example['trailing_whitespace'])
        labels_split_list.append(example['labels'])

    for tokens_list,lables_list,trailing_whitespace_list in zip(
        tokens_split_list,labels_split_list,trailing_whitespace_split_list):

        text = []
        labels = []
        for t, l, ws in zip(
            tokens_list, lables_list, trailing_whitespace_list
        ):
            text.append(t)
            labels.extend([l] * len(t))

            if ws:
                text.append(" ")
                labels.append("O")

        # actual tokenization
        tokenized = tokenizer("".join(text), return_offsets_mapping=True)

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

        example['bertlabels'].append(token_labels)
        example['berttokenids'].append(tokenized['input_ids'])
        example['berttokenmask'].append(tokenized['attention_mask'])
        example['berttokentoken_type_ids'].append(tokenized['token_type_ids'])

    return example


def expanddataset(ds,if_train=True):
    """
        将dataset的 bertlabels,berttokenpos2orgtokenpos expand到列,返回新的dataset
    """
    df=ds.to_pandas()
    merge_list_key=[]
    # s1 = pd.DataFrame(df.pop('bertlabels').values.tolist(), 
    #       index=df.index).stack().rename('bertlabels').reset_index(level=1, drop=True)
    if if_train :
        merge_list_key=['berttokenids','berttokenmask','berttokentoken_type_ids','bertlabels']
    else:
        merge_list_key=['berttokenids','berttokenmask','berttokentoken_type_ids','token_map','offset_mapping']
    s_l=[]
    for i in merge_list_key:
        tmp_s= pd.DataFrame(df.pop(i).values.tolist(), 
                    index=df.index).stack().rename(i).reset_index(level=1, drop=True)
        s_l.append(tmp_s)
    df = df.join(pd.concat(s_l, axis=1))
    return df.reset_index(drop=True)

def logit2truepredic(batch_predictions,batch_org_len):
    """
        按顺序返回列表
    """

    preds_final=[]
    batch_len=max(batch_org_len)
    for i,l in enumerate(batch_org_len):
        predictions=batch_predictions[i*batch_len:i*batch_len+l]

        softmaxed_pred=np.exp(predictions) / np.sum(np.exp(predictions), axis = 1).reshape(-1,1)
        preds = predictions.argmax(-1)
        preds_without_O = softmaxed_pred[:,:12].argmax(-1)
        O_preds = predictions[:,12]
        preds_final.append( list(np.where(O_preds < Config.threshold, preds_without_O , preds)))


    return preds_final