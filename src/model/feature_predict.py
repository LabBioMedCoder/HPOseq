import json
from evaluation import get_results
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, random_split, TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, \
    average_precision_score
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
              0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
              0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
              0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
              0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
              0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
              0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


def CalculateKSCTriad(sequence, gap, features, AADict):
	res = []
	for g in range(gap+1):
		myDict = {}
		for f in features:
			myDict[f] = 0

		for i in range(len(sequence)):
			if i+gap+1 < len(sequence) and i+2*gap+2<len(sequence):
				fea = AADict[sequence[i]] + '.' + AADict[sequence[i+gap+1]]+'.'+AADict[sequence[i+2*gap+2]]
				myDict[fea] = myDict[fea] + 1

		maxValue, minValue = max(myDict.values()), min(myDict.values())
		for f in features:
			res.append((myDict[f] - minValue) / maxValue)

	return res

def CTriad(fastas, gap = 0, **kw):
	AAGroup = {
		'g1': 'AGV',
		'g2': 'ILFP',
		'g3': 'YMTS',
		'g4': 'HNQW',
		'g5': 'RK',
		'g6': 'DE',
		'g7': 'C'
	}

	myGroups = sorted(AAGroup.keys())

	AADict = {}
	for g in myGroups:
		for aa in AAGroup[g]:
			AADict[aa] = g

	features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

	encodings = []
	header = ['#']
	for f in features:
		header.append(f)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		if len(sequence) < 3:
			print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
			return 0
		code = code + CalculateKSCTriad(sequence, 0, features, AADict)
		encodings.append(code)

	return encodings

def encode_protein_sequences_no_itertools(protein_sequences):
    # Define the 20 common amino acids
    amino_acids = 'AGVILFPYMTSHNQWRKDEC'

    # Generate a trinucleotide dictionary without using itertools
    trinucleotide_dict = {}
    idx = 0
    for a1 in amino_acids:
        for a2 in amino_acids:
            for a3 in amino_acids:
                trinucleotide_dict[f'{a1}{a2}{a3}'] = idx
                idx += 1
    trinucleotide_dict['000'] = idx
    # print(trinucleotide_dict)

    # Function to encode a single protein sequence
    def encode_sequence(sequence):
        # Remove non-common amino acids and adjust length
        cleaned_sequence = ''.join([aa for aa in sequence if aa in amino_acids])
        # Ensure the sequence length is 2000
        if len(cleaned_sequence) >= 2000:
            cleaned_sequence = cleaned_sequence[:2000]
        else:
            cleaned_sequence += '0' * (2000 - len(cleaned_sequence))

        # Encode the sequence
        encoded_sequence = [trinucleotide_dict.get(cleaned_sequence[i:i + 3], trinucleotide_dict['000'])
                            for i in range(2000 - 2)]

        return encoded_sequence

    # Encode all sequences
    encoded_sequences = [encode_sequence(seq) for seq in protein_sequences]

    return encoded_sequences

def calculate_performance(actual, pred_prob, threshold=0.4, average='micro'):
    pred_lable = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]) > threshold).astype(np.int)
        eachline = eachline.tolist()
        pred_lable.append(eachline)
    f_score = f1_score(np.array(actual), np.array(pred_lable), average=average)
    recall = recall_score(np.array(actual), np.array(pred_lable), average=average)
    precision = precision_score(np.array(actual), np.array(pred_lable), average=average)
    return f_score, recall, precision

class CNN_Module(nn.Module):
    def __init__(self, input_data, hidden_size):
        super(CNN_Module, self).__init__()
        self.input_dim = input_data

        # Embedding layer
        self.embedding = nn.Embedding(8001, 128).to(device)  # 8001 to include '000' and padding index
        # 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=64, padding=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=32, padding=1).to(device)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=16, padding=1).to(device)
        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=64, stride=32).to(device)
        # Fully connected layers
        self.fc1 = nn.Linear(928, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, 4575).to(device)
        # Activation function
        self.relu = nn.ReLU().to(device)
        self.relu = nn.ReLU().to(device)
        # 1D Convolutional layer

    def forward(self, x):
        x = self.embedding(x)  # Embedding
        x = x.permute(0, 2, 1)  # Rearrange dimensions for Conv1d
        x = F.relu(self.conv1(x))  # Convolutional layer 1
        x = F.relu(self.conv2(x))  # Convolutional layer 2
        x = F.relu(self.conv3(x))  # Convolutional layer 3
        x = self.pool(x)  # Max pooling
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)  # Fully connected layer 1
        x = F.sigmoid(self.fc2(x))  # Fully connected layer 2 (Output layer)
        return x

def Evaluation_result(predict_result, hpo_list, feature_name):
    Y_test = hpo_list
    y_score = predict_result
    perf_cc = get_results(Y_test, y_score)
    print(feature_name,":",perf_cc['all'])

def CNN_model_train(learningrate, batchsize, epochtime, hidden_size, seq_list, hpo_list):
    batch_size = batchsize
    learning_rate = learningrate
    epoch_times = epochtime
    loss_function = nn.BCELoss()

    fuse_list = encode_protein_sequences_no_itertools(seq_list)

    old_fuse_list = fuse_list
    feature_len = len(fuse_list[0])
    print("输入节点数为：", feature_len)
    # 定义划分比例
    train_ratio = 0.8
    # 计算划分的数量
    train_size = int(train_ratio * len(fuse_list))
    test_size = len(fuse_list) - train_size
    # 将数据和标签构成torch数据集
    hpos = torch.tensor(hpo_list, dtype=torch.float)
    fuse_list = torch.tensor(fuse_list, dtype=torch.int)
    dataset = TensorDataset(fuse_list, hpos)
    # 随机划分数据集
    train_benchmark, test_benchmark = random_split(dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_benchmark, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_benchmark, batch_size=batch_size, shuffle=False)

    cnn_model = CNN_Module(feature_len, hidden_size).to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=0.00001)
    cnn_model.train()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        for batch_idx, data in enumerate(train_data_loader):
            feature = data[0]
            hpo = data[1]
            feaVect = Variable(feature).to(device)
            HPO_annotiations = torch.squeeze(hpo)
            HPO_annotiations = Variable(HPO_annotiations).to(device)
            out = cnn_model(feaVect)
            optimizer.zero_grad()
            loss = loss_function(out, HPO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.data
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, data in enumerate(test_data_loader):
            ppiVect = data[0]
            hpos = data[1]
            ppiVect = Variable(ppiVect).to(device)
            HPO_annotiations = Variable(hpos).to(device)
            out = cnn_model(ppiVect)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(HPO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, HPO_annotiations)
            t_loss += one_loss.data
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0
        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(cnn_model,
                       '../../data/nn/CNNVal_{}_{}_{}.pkl'.format(batch_size, learning_rate,
                                                                                  epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))

#     test_PPImodel = torch.load(
#         '../../data/nn/CNNVal_{}_{}_{}.pkl'.format(batch_size, learning_rate, epoch_times)).to(device)
#     t_loss = 0
#     ppi_test_outs = {}
#     pred = []
#     actual = []
#     score_dict = {}
#     batch_num = 0
#     for batch_idx, data in enumerate(test_data_loader):
#         ppiVect = data[0]
#         hpos = data[1]
#         ppiVect = Variable(ppiVect).to(device)
#         HPO_annotiations = Variable(hpos).to(device)
#         out = test_PPImodel(ppiVect)
#         batch_num += 1
#         ppi_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
#         pred.append(out.data[0].cpu().tolist())
#         actual.append(HPO_annotiations.data[0].cpu().tolist())
#         loss = loss_function(out, HPO_annotiations)
#         t_loss += loss.data
#     test_loss = "{}".format(t_loss / batch_num)
#     fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
#     auc_score = auc(fpr, tpr)
#     each_best_fcore = 0
#     for i in range(len(Thresholds)):
#         f_score, recall, precision = calculate_performance(
#             actual, pred, threshold=Thresholds[i], average='micro')
#         if f_score > each_best_fcore:
#             each_best_fcore = f_score
#             each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
#         scores = [f_score, recall, precision, auc_score]
#         score_dict[Thresholds[i]] = scores
#     bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
#     prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]

#     print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
#         test_loss, learning_rate, batch_size, epoch_times,
#         f_max, auc_score, recall_max, prec_max, bestthreshold))

    prediction_CNNmodel = torch.load(
        '../../data/nn/CNNVal_{}_{}_{}.pkl'.format(batch_size, learning_rate, epoch_times)).to(device)
    out_dict = {}
    new_fuse_list = []
    for feature in fuse_list:
        feaMatrix = torch.tensor(feature, dtype=torch.int)
        feaMatrix = feaMatrix.unsqueeze(0)
        feaMatrix = Variable(feaMatrix).to(device)
        out = prediction_CNNmodel(feaMatrix)
        new_fuse_list.append(out.data[0].cpu().tolist())

    return new_fuse_list  # 返回再最优的Seq模型下的训练集的输出和测试集的输出，用于训练weight_classifier

def VGAE_model_train(learningrate, batchsize, epochtime, hidden_size, seq_list, adj, hpo_list):
    batch_size = batchsize
    learning_rate = learningrate
    epoch_times = epochtime
    loss_function = nn.BCELoss()

    fastas = []
    index = 1
    for seq in seq_list:
        temp_tuple = (str(index),seq)
        fastas.append(temp_tuple)
        index+=1
    CT_encoded_sequences = CTriad(fastas)
    feature_list = []
    for CT in CT_encoded_sequences[1:]:
        feature_list.append(CT[1:])



#------------------------------------------------------------------------------------------------------------------
    fuse_list = encode_protein_sequences_no_itertools(seq_list)

    old_fuse_list = fuse_list
    feature_len = len(fuse_list[0])
    print("输入节点数为：", feature_len)
    # 定义划分比例
    train_ratio = 0.8
    # 计算划分的数量
    train_size = int(train_ratio * len(fuse_list))
    test_size = len(fuse_list) - train_size
    # 将数据和标签构成torch数据集
    hpos = torch.tensor(hpo_list, dtype=torch.float)
    fuse_list = torch.tensor(fuse_list, dtype=torch.int)
    dataset = TensorDataset(fuse_list, hpos)
    # 随机划分数据集
    train_benchmark, test_benchmark = random_split(dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_benchmark, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_benchmark, batch_size=batch_size, shuffle=False)

    cnn_model = CNN_Module(feature_len, hidden_size).to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=0.00001)
    cnn_model.train()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        for batch_idx, data in enumerate(train_data_loader):
            feature = data[0]
            hpo = data[1]
            feaVect = Variable(feature).to(device)
            HPO_annotiations = torch.squeeze(hpo)
            HPO_annotiations = Variable(HPO_annotiations).to(device)
            out = cnn_model(feaVect)
            optimizer.zero_grad()
            loss = loss_function(out, HPO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.data
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, data in enumerate(test_data_loader):
            ppiVect = data[0]
            hpos = data[1]
            ppiVect = Variable(ppiVect).to(device)
            HPO_annotiations = Variable(hpos).to(device)
            out = cnn_model(ppiVect)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(HPO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, HPO_annotiations)
            t_loss += one_loss.data
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0
        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(cnn_model,
                       '../../data/nn/CNNVal_{}_{}_{}.pkl'.format(batch_size, learning_rate,
                                                                                  epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))

#     test_PPImodel = torch.load(
#         '../../data/nn/CNNVal_{}_{}_{}.pkl'.format(batch_size, learning_rate, epoch_times)).to(device)
#     t_loss = 0
#     ppi_test_outs = {}
#     pred = []
#     actual = []
#     score_dict = {}
#     batch_num = 0
#     for batch_idx, data in enumerate(test_data_loader):
#         ppiVect = data[0]
#         hpos = data[1]
#         ppiVect = Variable(ppiVect).to(device)
#         HPO_annotiations = Variable(hpos).to(device)
#         out = test_PPImodel(ppiVect)
#         batch_num += 1
#         ppi_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
#         pred.append(out.data[0].cpu().tolist())
#         actual.append(HPO_annotiations.data[0].cpu().tolist())
#         loss = loss_function(out, HPO_annotiations)
#         t_loss += loss.data
#     test_loss = "{}".format(t_loss / batch_num)
#     fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
#     auc_score = auc(fpr, tpr)
#     each_best_fcore = 0
#     for i in range(len(Thresholds)):
#         f_score, recall, precision = calculate_performance(
#             actual, pred, threshold=Thresholds[i], average='micro')
#         if f_score > each_best_fcore:
#             each_best_fcore = f_score
#             each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
#         scores = [f_score, recall, precision, auc_score]
#         score_dict[Thresholds[i]] = scores
#     bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
#     prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]

#     print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
#         test_loss, learning_rate, batch_size, epoch_times,
#         f_max, auc_score, recall_max, prec_max, bestthreshold))

    prediction_CNNmodel = torch.load(
        '../../data/nn/CNNVal_{}_{}_{}.pkl'.format(batch_size, learning_rate, epoch_times)).to(device)
    out_dict = {}
    new_fuse_list = []
    for feature in fuse_list:
        feaMatrix = torch.tensor(feature, dtype=torch.int)
        feaMatrix = feaMatrix.unsqueeze(0)
        feaMatrix = Variable(feaMatrix).to(device)
        out = prediction_CNNmodel(feaMatrix)
        new_fuse_list.append(out.data[0].cpu().tolist())

    return new_fuse_list  # 返回再最优的Seq模型下的训练集的输出和测试集的输出，用于训练weight_classifier



uniprot = pd.read_pickle('../../data/features_20211010.pkl')
print(uniprot.columns)

hpo_list = []
seq_list = []
# pro_dict = {}
# index = 0
for i, row in uniprot.iterrows():
    hpo_label = row['hpo_one_hot']
    seq = row['seq']
    hpo_list.append(hpo_label)
    seq_list.append(seq)
    # pro_name = row['From']
    # pro_dict[pro_name] = str(index)
    # index+=1

adj = load_network("../../data/network/blast_network.txt", uniprot.shape[0])

seq_predict = CNN_model_train(0.001, 512, 1, 2048, seq_list, hpo_list)  # 80
# net_predict = VGAE_model_train(0.001, 512, 80, 2048, seq_list, adj, hpo_list)  # 80
print(seq_predict)
Evaluation_result(seq_predict, hpo_list, "seq_result")

uniprot['seq_result'] = seq_predict

uniprot.to_pickle("../../data/features_20211010.pkl")
