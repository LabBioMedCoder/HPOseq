import pandas as pd
import re
from GO_pre import get_gene_ontology, get_anchestors
import numpy as np
import argparse
import json

def get_phenotype_ontology(filename='../../data/HPO/hp_20211010.txt'):
    # Reading Gene Ontology from OBO Formatted file
    hp = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    hp[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        hp[obj['id']] = obj
    for hp_id in list(hp.keys()):
        if hp[hp_id]['is_obsolete']:
            del hp[hp_id]
    for hp_id, val in hp.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in hp:
                if 'children' not in hp[p_id]:
                    hp[p_id]['children'] = set()
                hp[p_id]['children'].add(hp_id)
    return hp

def statistic_seq_len(list_2d):
    len_0_500=0
    len_501_1000 = 0
    len_1001_1500 = 0
    len_all = 0
    for a_seq in list_2d:
        if len(a_seq) <= 500:
            len_0_500+=1
        elif len(a_seq) <= 1000:
            len_501_1000+=1
        elif len(a_seq) <= 1500:
            len_1001_1500+=1
        len_all+=len(a_seq)
    ave_len = len_all/len(list_2d)
    return len_0_500,len_501_1000,len_1001_1500,ave_len

def statistic_HPO_frequency(HPO_num_dict):
    len_0_9=0
    len_10_49 = 0
    len_50_99 = 0
    len_100 = 0
    len_all = 0
    for hpo in HPO_num_dict:
        if HPO_num_dict[hpo] <= 9:
            len_0_9+=1
        elif HPO_num_dict[hpo] <= 49:
            len_10_49+=1
        elif HPO_num_dict[hpo] <= 99:
            len_50_99+=1
        elif HPO_num_dict[hpo] > 99:
            len_100+=1
        len_all+=HPO_num_dict[hpo]
    ave_len = len_all/len(list(HPO_num_dict.keys()))
    return len_0_9,len_10_49,len_50_99,len_100,ave_len


with open("../../data/HPO/gene_hpo.json") as fp:
    # with open("../../data/20200825_new/hpo_annotation_raw_20200825.json") as fp:
    gene_hpo_dict = json.load(fp)
uniprot = pd.read_excel('../../data/HPO/uniprot.xlsx')

gene_pro_dict = {}
pro_hpo_dict = {}
for i, row in uniprot.iterrows():
    gene_id = row['From']
    pro_name = row['Entry']
    if gene_id not in gene_pro_dict:
        gene_pro_dict[gene_id] = []
    gene_pro_dict[gene_id].append(pro_name)


for gene in gene_pro_dict:
    for pro in gene_pro_dict[gene]:
        pro_hpo_dict[pro] = gene_hpo_dict[str(gene)]

pro_info = []
seq_list = []
for ind,row in uniprot.iterrows():
    pro_name = row['Entry']
    pro_info.append(pro_name)

print("蛋白质数量：",len(pro_info))

for ind,row in uniprot.iterrows():
    pro_name = row['Entry']
    seq_info = row['Sequence']
    if pro_name in pro_info:
        seq_list.append(seq_info)

hp = get_phenotype_ontology()

new_hp = []
for pro in pro_info:
    labels = pro_hpo_dict[pro]
    # print("labels",len(labels))
    temp = set([])
    for x in labels:
        temp = temp | get_anchestors(hp, x)
    # print("temp",len(temp))
    new_hp.append(list(temp))
print(len(new_hp))


num_hp = 0
for pro in pro_info:
    num_hp+=len(pro_hpo_dict[pro])
num_hp_after = sum(len(x) for x in new_hp)
print("原有HPO数目",num_hp)
print("新HPO数目",num_hp_after)
HPO_list = []
for list_info in new_hp:
    for hpo in list_info:
        if hpo not in HPO_list:
            HPO_list.append(hpo)
print("HPO标签：",len(HPO_list))

#去除其中注释频率小于10的hpo注释
HPO_list_num_dict = {}
for a in HPO_list:
    HPO_list_num_dict[a]=0
for list_info in new_hp:
    for hpo in list_info:
        if hpo in HPO_list:
            HPO_list_num_dict[hpo] = HPO_list_num_dict[hpo]+1

for hpo_name in HPO_list_num_dict.keys():
    if HPO_list_num_dict[hpo_name]<10:
        HPO_list.remove(hpo_name)
print("删去频率在10以下的后，HPO标签：",len(HPO_list))

flag_new_hpo_list = []
pro_hpo_true_dict = {}
hpo_pro_true_dict = {}
for pro in pro_hpo_dict:
    if pro in pro_info:
        temp_list = []
        for hpo in pro_hpo_dict[pro]:
            if hpo in HPO_list:
                temp_list.append(hpo)
        pro_hpo_true_dict[pro] = temp_list

for pro in pro_hpo_true_dict:
    for hpo in pro_hpo_dict[pro]:
        if hpo not in hpo_pro_true_dict:
            hpo_pro_true_dict[hpo] = []
        hpo_pro_true_dict[hpo].append(pro)


print("编码HPO项目...")
hp_dict = dict(zip(HPO_list, range(len(HPO_list))))
hp_encoding = [[0] * len(HPO_list) for i in pro_info]

i = 0
for pro in pro_info:
    for x in pro_hpo_true_dict[pro]:
        if x in hp_dict:
            hp_encoding[i][hp_dict[x]] = 1
    i += 1

pro_hpo_list = []
for pro in pro_info:
    pro_hpo_list.append(pro_hpo_true_dict[pro])

print(len(pro_info),len(seq_list),len(pro_hpo_list),len(hp_encoding))
temp_dict = {'From':pro_info,'seq':seq_list,'hpo':pro_hpo_list,'hpo_one_hot':hp_encoding}
df = pd.DataFrame(temp_dict)

hp_encoding_dict = {}
num_dict = 0
for pro in pro_info:
    hp_encoding_dict[pro] = hp_encoding[num_dict]
    num_dict += 1

a,b,c,ave = statistic_seq_len(seq_list)
print(a,b,c,ave)
print("#删去出现次数小于10的HPO列表：", str(len(HPO_list)))

# with open('hpolist.txt',"w") as fptxt:
#     fptxt.write("#删去出现次数小于10的HPO列表："+ str(len(HPO_list)) + "\n")
#     for hpo in HPO_list:
#         fptxt.write(hpo+"\n")

# with open("pro_to_hpo_tl.json", "w") as f:
#     json.dump(pro_hpo_true_dict, f)
#
# with open("hpo_to_pro_tl.json", "w") as f:
#     json.dump(hpo_pro_true_dict, f)
#
# with open("pro_to_hpo_tl_embedding.json", "w") as f:
#     json.dump(hp_encoding_dict, f)

with open('../../data/HPO/prolist.txt',"w") as fptxt:
    fptxt.write("#PRO数量："+ str(len(pro_info)) + "\n")
    print("#PRO数量：",str(len(pro_info)))
    for pro in pro_info:
        fptxt.write(pro+"\n")

print(df.columns)
df.to_pickle("../../data/features_20211010.pkl")
df.to_excel('../../data/features_20211010.xlsx')

