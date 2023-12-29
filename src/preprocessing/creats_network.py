#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pandas as pd

pro_dict = {}
uniprot = pd.read_pickle('../../data/features_20211010.pkl')
index = 0
for i, row in uniprot.iterrows():
    pro_name = row['From']
    pro_dict[pro_name] = index
    index+=1
print(pro_dict)

network_dict = {}
thresholds = 0.01
with open("../../data/network/blastp_list.txt") as fp:
    for line in fp:
        if line.startswith('#'):
            continue
        # print(line)
        protein1,protein2,_,_,_,_,_,_,_,_,value,_= line.strip().split('\t')
        if protein1 not in network_dict:
            network_dict[protein1] = {}
        if protein1!=protein2 and float(value)<=thresholds and protein2 not in network_dict[protein1] :
            network_dict[protein1][protein2] = 1-float(value)
print(network_dict['A0PJY2'])

net_list = []
for pro1 in network_dict:
    for pro2 in network_dict[pro1]:
        str_info = str(pro_dict[pro1])+" "+str(pro_dict[pro2])+" "+ str(network_dict[pro1][pro2])
        net_list.append(str_info)

with open('../../data/network/blast_network.txt',"w") as fptxt:
    for relation in net_list:
        fptxt.write(relation+"\n")

