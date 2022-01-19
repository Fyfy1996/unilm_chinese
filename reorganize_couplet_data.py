# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:33:49 2022

@author: test1
"""

data_path = "data/couplet/test/"

f1 = open(data_path+"in.txt","r",encoding="utf8")
f2 = open(data_path+"out.txt","r",encoding="utf8")


inputs = f1.readlines()
ouputs = f2.readlines()

f1.close()
f2.close()


#%%
import json
with open("data/couplet/test.json","w", encoding="utf8") as f:
    for i in range(len(inputs)):
        tmp = json.dumps({
                "src_text": inputs[i].strip().replace(" ",""),
                "tgt_text": ouputs[i].strip().replace(" ","")
            },ensure_ascii=False)
        f.write(tmp)
        f.write("\n")