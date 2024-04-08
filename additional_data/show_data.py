import json
import numpy as np


def show_data(name):
    
    print(f"==================================== {name} ===================================")
    label_num = {"B-EMAIL": 0, "B-ID_NUM": 0, "B-NAME_STUDENT": 0, "B-PHONE_NUM": 0, "B-STREET_ADDRESS": 0, "B-URL_PERSONAL": 0, "B-USERNAME": 0, "I-ID_NUM": 0, "I-NAME_STUDENT": 0, "I-PHONE_NUM": 0, "I-STREET_ADDRESS": 0, "I-URL_PERSONAL": 0, "O": 0}
    with open(name) as f:
        data = json.load(f)

    doc_len = []
    total_label, full_O = 0, 0
    for i in data:
        doc_len.append(len(i['tokens']))
        for j in i['labels']:
            label_num[j] += 1
            total_label += 1
        if not any(np.array(i["labels"]) != "O"): 
            full_O += 1
    


    print(f"doc num: {len(data)}, min lengh: {min(doc_len)}, avg minlengh: {np.mean(doc_len)}, median lengh: {np.median(doc_len)}, max lengh: {max(doc_len)}")
    for idx, (k,v) in enumerate(label_num.items()):
        if idx % 6 == 0 and idx > 0:
            print(f"{k}: {v}",end='\n')
        else :
            print(f"{k}: {v}       ", end="")
    
    print(f"total label: {total_label}, O proportion: {label_num['O']/total_label}, full O num: {full_O}")
        
    print()




if __name__ == '__main__':
    name = ["train.json",'mixtral-8x7b-v1.json','mpware_mixtral8x7b_v1.1-no-i-username.json', \
            'pii_dataset_fixed.json','moredata_dataset_fixed.json','mpware_mixtral_clean.json',\
            'mixtral-8x7b-v1_clean.json','moredata_dataset_clean.json','moredata_dataset_clean2.json',\
            'mpware_clean2.json'    ]
    for n in name:
        show_data(n)

