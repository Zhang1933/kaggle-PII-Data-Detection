from cfg import *

if not Config.split_by_paragraph:
    from nltk.tokenize import sent_tokenize


from unidecode import unidecode
import copy
from tqdm import tqdm,tqdm_notebook

# batch 对齐
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
def split_document(text):
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

def preprocesss(example,tokenizer,label2id,if_train=True):
    """
        预处理
        需要特判token为空字符的标签情况。需要特判berttoken多对一映射token情况,联系前后token判断一下就行,if if_train =true,all bertlabels=-100
    """
    # unicode to ascii
    example['full_text']=unidecode(example['full_text'],errors='replace',replace_str='*')
    example['full_text']=example['full_text'].lower()
    for j in range(len(example['tokens'])):
        example['tokens'][j]=unidecode(example['tokens'][j],errors='replace',replace_str='*')
        example['tokens'][j]=example['tokens'][j].lower().strip()

    # split text
    example['splited_sens']=split_document(example['full_text'])
    
    example['berttokenpos2orgtokenpos']=[]
    example['bertlabels']=[]
    example['berttokenids']=[]
    example['berttokenmask']=[]
    example['berttokentoken_type_ids']=[]
    
    # tokenize splited_sen
    previs_sep=0
    org_token_id=0
    for splited_sen in example['splited_sens']:
        out=tokenizer(splited_sen)
        splited_strtokens=tokenizer.convert_ids_to_tokens(out['input_ids'])
        splited_token_len=len(splited_strtokens)
        berttoken_pos2orgtokenpos=[-1]*splited_token_len
        bertlabel=[-100]*splited_token_len        
        find_start=0
        splited_token_id=1
        nothit=0 # 连续多次没映射到说明有问题
        while splited_token_id < splited_token_len-1: # 不处理开头与末尾处的[CLS]与[SEP]
            if splited_strtokens[splited_token_id]=='[SEP]':
                bertlabel[splited_token_id]=-100
                berttoken_pos2orgtokenpos[splited_token_id]=-1
                if  splited_token_id!=1:
                    previs_sep=org_token_id
                else:
                    org_token_id=previs_sep
                splited_token_id+=1
                continue
            subtokenstr=splited_strtokens[splited_token_id]
            if(splited_strtokens[splited_token_id][0]=='▁') and len(splited_strtokens[splited_token_id])!=1:
                    subtokenstr=subtokenstr[1:]
            # find start of the substring
            find_start=example['tokens'][org_token_id].find(subtokenstr,find_start)
            if find_start != -1:
                berttoken_pos2orgtokenpos[splited_token_id]=org_token_id
                if if_train: bertlabel[splited_token_id]=label2id[example['labels'][org_token_id]]
                find_start+=len(subtokenstr)
                nothit=0
            else:
                nothit+=1
                if nothit>3:
                    print("Warning:failed to hit multiple times in a row,nothit:{},doc_id:{}".format(nothit,example['document'])) # 方便调试
            if example['tokens'][org_token_id]:
                if find_start==-1:
                    # example['tokens'][org_token_id]='can',subtokenstr='cannot' 的情况
                    tmp_find=0
                    berttoken_pos2orgtokenpos[splited_token_id]=org_token_id # 只记录第一个label
                    if if_train: bertlabel[splited_token_id]=label2id[example['labels'][org_token_id]]
                    while subtokenstr.find(example['tokens'][org_token_id],tmp_find)!=-1:
                        tmp_find+=len(example['tokens'][org_token_id])
                        org_token_id+=1
                    org_token_id-=1
                splited_token_id+=1
            if(find_start==-1 or find_start==len(example['tokens'][org_token_id])):
                org_token_id+=1
                find_start=0
        example['berttokenids'].append(out['input_ids'])
        example['berttokenmask'].append(out['attention_mask'])
        example['berttokentoken_type_ids'].append(out['token_type_ids'])
        example['berttokenpos2orgtokenpos'].append(berttoken_pos2orgtokenpos)
        example['bertlabels'].append(bertlabel)
    return example


def expanddataset(ds):
    """
        将dataset的 bertlabels,berttokenpos2orgtokenpos expand到列,返回新的dataset
    """
    df=ds.to_pandas()
    merge_list_key=['berttokenpos2orgtokenpos','berttokenids','berttokenmask','berttokentoken_type_ids','bertlabels']
    # s1 = pd.DataFrame(df.pop('bertlabels').values.tolist(), 
    #       index=df.index).stack().rename('bertlabels').reset_index(level=1, drop=True)

    s_l=[]
    for i in merge_list_key:
        tmp_s= pd.DataFrame(df.pop(i).values.tolist(), 
                    index=df.index).stack().rename(i).reset_index(level=1, drop=True)
        s_l.append(tmp_s)
    df = df.join(pd.concat(s_l, axis=1))
    return df.reset_index(drop=True)

def logit2truepredic(predictions):
    softmaxed_pred=np.exp(predictions) / np.sum(np.exp(predictions), axis = 1).reshape(-1,1)
    preds = predictions.argmax(-1)
    preds_without_O = softmaxed_pred[:,:12].argmax(-1)
    O_preds = predictions[:,12]
    preds_final = np.where(O_preds < Config.threshold, preds_without_O , preds)
    return preds_final