#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

geneid = []
gene_hpo_dict = {}



with open("../../data/HPO/phenotype_to_genes.txt") as fp:
    for line in fp:
        if line.startswith('#'):
            continue
        # print(line)
        hpo,_,gene_id,_,_,_,_= line.strip().split('\t')
        if gene_id not in gene_hpo_dict:
            gene_hpo_dict[gene_id] = []
        if hpo not in gene_hpo_dict[gene_id]:
            gene_hpo_dict[gene_id].append(hpo)
        if gene_id not in geneid:
            geneid.append(gene_id)

# for gene in gene_hpo_dict:
#     new_list = gene_hpo_dict[gene]
#     print(new_list)
#     gene_hpo_dict[gene] = line(set(new_list))

print(gene_hpo_dict)

with open('../../data/HPO/genelist.txt',"w") as fptxt:
    fptxt.write("#HPO注释对应基因列表："+ str(len(geneid)) + "\n")
    for gene in geneid:
        fptxt.write(gene+"\n")

with open("../../data/HPO/gene_hpo.json", "w") as f:
    json.dump(gene_hpo_dict, f)
