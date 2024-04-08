# PII 模型说明

1. 下载权重和数据集

2. 执行命令
```sh
python ./train.py
```

3. 将model_weight中的权重上传至kaggle平台测评，其中平台代码可以采用inference.ipynb文件，仅需修改第一个代码块的路径即可

# 增量说明

1. 采用更小的学习率(1e-5)，目的是减小deberta权重变化增加泛化性
2. 使用 lstm + linear 分类
3. 新增加数据集，采用更多的数据集训练和测试
4. 增加epoch 模型大概在第四轮左右开始出现过拟合

注意：
1. 引入新数据集时不要引入全'O'文本
2. 测试集精度并不能估计模型真实精度
3. 相同的方式训练模型的，精度可能存在差距，可能也是难找到最充分训练的位置，估计在2%



# 数据集特征

```
==================================== train.json ===================================
doc num: 6807, min lengh: 69, avg minlengh: 733.4410166005582, median lengh: 693.0, max lengh: 3298
B-EMAIL: 39       B-ID_NUM: 78       B-NAME_STUDENT: 1365       B-PHONE_NUM: 6       B-STREET_ADDRESS: 2       B-URL_PERSONAL: 110       B-USERNAME: 6
I-ID_NUM: 1       I-NAME_STUDENT: 1096       I-PHONE_NUM: 15       I-STREET_ADDRESS: 20       I-URL_PERSONAL: 1       O: 4989794
total label: 4992533, O proportion: 0.9994513806919253, full O num: 5862

==================================== mixtral-8x7b-v1.json ===================================
doc num: 2355, min lengh: 40, avg minlengh: 878.2335456475583, median lengh: 861.0, max lengh: 1846
B-EMAIL: 2448       B-ID_NUM: 2376       B-NAME_STUDENT: 5937       B-PHONE_NUM: 2388       B-STREET_ADDRESS: 2294       B-URL_PERSONAL: 3297       B-USERNAME: 2374
I-ID_NUM: 1059       I-NAME_STUDENT: 7459       I-PHONE_NUM: 7974       I-STREET_ADDRESS: 19422       I-URL_PERSONAL: 0       O: 2011212
total label: 2068240, O proportion: 0.9724267976637141, full O num: 0

==================================== mpware_mixtral8x7b_v1.1-no-i-username.json ===================================
doc num: 2692, min lengh: 17, avg minlengh: 784.7492570579495, median lengh: 776.0, max lengh: 1349
B-EMAIL: 870       B-ID_NUM: 1033       B-NAME_STUDENT: 4614       B-PHONE_NUM: 817       B-STREET_ADDRESS: 799       B-URL_PERSONAL: 1238       B-USERNAME: 841
I-ID_NUM: 1240       I-NAME_STUDENT: 5866       I-PHONE_NUM: 2530       I-STREET_ADDRESS: 6219       I-URL_PERSONAL: 0       O: 2086478
total label: 2112545, O proportion: 0.987660854561678, full O num: 369

==================================== pii_dataset_fixed.json ===================================
doc num: 4434, min lengh: 98, avg minlengh: 354.1321605773568, median lengh: 355.0, max lengh: 609
B-EMAIL: 3730       B-ID_NUM: 0       B-NAME_STUDENT: 11055       B-PHONE_NUM: 1603       B-STREET_ADDRESS: 3510       B-URL_PERSONAL: 582       B-USERNAME: 670
I-ID_NUM: 0       I-NAME_STUDENT: 5597       I-PHONE_NUM: 2632       I-STREET_ADDRESS: 8771       I-URL_PERSONAL: 0       O: 1532072
total label: 1570222, O proportion: 0.9757040724177856, full O num: 10

==================================== moredata_dataset_fixed.json ===================================
doc num: 2000, min lengh: 181, avg minlengh: 509.637, median lengh: 503.0, max lengh: 1228
B-EMAIL: 2603       B-ID_NUM: 1958       B-NAME_STUDENT: 22703       B-PHONE_NUM: 2589       B-STREET_ADDRESS: 2591       B-URL_PERSONAL: 2582       B-USERNAME: 2694
I-ID_NUM: 288       I-NAME_STUDENT: 5368       I-PHONE_NUM: 6980       I-STREET_ADDRESS: 21531       I-URL_PERSONAL: 0       O: 947387
total label: 1019274, O proportion: 0.9294723499274974, full O num: 1

==================================== mpware_mixtral_clean.json ===================================
doc num: 1456, min lengh: 17, avg minlengh: 776.7630494505495, median lengh: 776.0, max lengh: 1342
B-EMAIL: 200       B-ID_NUM: 214       B-NAME_STUDENT: 2377       B-PHONE_NUM: 159       B-STREET_ADDRESS: 169       B-URL_PERSONAL: 469       B-USERNAME: 168
I-ID_NUM: 0       I-NAME_STUDENT: 3141       I-PHONE_NUM: 516       I-STREET_ADDRESS: 1278       I-URL_PERSONAL: 0       O: 1122276
total label: 1130967, O proportion: 0.9923154256490242, full O num: 329

==================================== mixtral-8x7b-v1_clean.json ===================================
doc num: 1967, min lengh: 97, avg minlengh: 881.1138790035587, median lengh: 864.0, max lengh: 1846
B-EMAIL: 1980       B-ID_NUM: 1908       B-NAME_STUDENT: 4922       B-PHONE_NUM: 1832       B-STREET_ADDRESS: 1706       B-URL_PERSONAL: 2720       B-USERNAME: 1923
I-ID_NUM: 817       I-NAME_STUDENT: 6164       I-PHONE_NUM: 6253       I-STREET_ADDRESS: 14361       I-URL_PERSONAL: 0       O: 1688565
total label: 1733151, O proportion: 0.9742746015782814, full O num: 0

==================================== moredata_dataset_clean.json ===================================
doc num: 313, min lengh: 247, avg minlengh: 479.1309904153355, median lengh: 475.0, max lengh: 762
B-EMAIL: 314       B-ID_NUM: 329       B-NAME_STUDENT: 3942       B-PHONE_NUM: 285       B-STREET_ADDRESS: 336       B-URL_PERSONAL: 382       B-USERNAME: 438
I-ID_NUM: 0       I-NAME_STUDENT: 708       I-PHONE_NUM: 1028       I-STREET_ADDRESS: 2817       I-URL_PERSONAL: 0       O: 139389
total label: 149968, O proportion: 0.9294582844340126, full O num: 0

==================================== moredata_dataset_clean2.json ===================================
doc num: 29, min lengh: 383, avg minlengh: 500.2758620689655, median lengh: 489.0, max lengh: 644
B-EMAIL: 27       B-ID_NUM: 24       B-NAME_STUDENT: 387       B-PHONE_NUM: 26       B-STREET_ADDRESS: 22       B-URL_PERSONAL: 33       B-USERNAME: 31
I-ID_NUM: 0       I-NAME_STUDENT: 73       I-PHONE_NUM: 88       I-STREET_ADDRESS: 167       I-URL_PERSONAL: 0       O: 13630
total label: 14508, O proportion: 0.9394816652881169, full O num: 0

==================================== mpware_clean2.json ===================================
doc num: 1625, min lengh: 17, avg minlengh: 787.4627692307693, median lengh: 785.0, max lengh: 1349
B-EMAIL: 77       B-ID_NUM: 61       B-NAME_STUDENT: 2808       B-PHONE_NUM: 45       B-STREET_ADDRESS: 29       B-URL_PERSONAL: 388       B-USERNAME: 62
I-ID_NUM: 0       I-NAME_STUDENT: 3660       I-PHONE_NUM: 108       I-STREET_ADDRESS: 224       I-URL_PERSONAL: 0       O: 1272165
total label: 1279627, O proportion: 0.9941686131974395, full O num: 366
```