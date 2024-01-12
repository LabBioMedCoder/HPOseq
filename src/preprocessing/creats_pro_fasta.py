import pandas as pd

uniprot = pd.read_pickle('../../data/features_20211010.pkl')

new_pro_list = []
with open("../../data/HPO/prolist.txt") as fp:
    for line in fp:
        if line.startswith('#'):
            continue
        line = line.strip().split('\t')
        new_pro_list.append(line[0])

number = 0
# df = [4631 rows x 2 columns]
with open('../../data/HPO/protein.fasta', "w") as fp:
    for ind,row in uniprot.iterrows():
        pro_name = row['From']
        attribute = row['seq']
        fp.write(">" + pro_name + "\n")
        fp.write(attribute + "\n")
        number += 1
    fp.write("#蛋白质序列数量：" + str(number))
