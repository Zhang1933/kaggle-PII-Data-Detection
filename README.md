trainning: playground.ipynb

inference: inference.ipynb


|model id| 模型|  dataset&训练方式   | 分割方式 | training dataset validation | LB |
|-| ----------- |-| ----------- |-|-|
|0.991| deberta-base| baseline语料库训练  | split by token '\n\n' (上下文可能过短)    |0.951|[0.904](https://www.kaggle.com/code/xorspace/debert-base/notebook?scriptVersionId=162905902)|
|0.991| deberta-base   |    baseline语料库训练,滑动窗口(1024,8)  |训练by token，推理by windows|0.972||
|0.989| deberta-base   |    baseline语料库训练,滑动窗口(1024,16)  |训练推理都by windows|0.977|
