import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from src.model.evaluation import get_results
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask = mask

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias).to(device)


class Fuse_Model(nn.Module):
    def __init__(self, output_size,mask_ma1):
        super(Fuse_Model, self).__init__()
        self.weight_layer1 = MaskedLinear(output_size*2, output_size, mask_ma1).to(device)
        self.outlayer= nn.Linear(output_size, output_size).to(device)

    def forward(self, x):
        x = self.weight_layer1(x)
        x = F.relu(x)
        x = self.outlayer(x)
        x = F.sigmoid(x)
        return x

print("开始加载pkl...")
feature = []
CTDC = []
FAMILY = []
DOMAIN = []
PPMI = []
label = []
mask_info= []
uniprot = pd.read_pickle("data/features_20211010.pkl")
print(uniprot.columns)
feature_num = 2
for i, row in uniprot.iterrows():
    temp_list = []
    temp_list.extend(row['intra_result'])
    temp_list.extend(row['inter_result'])
    feature.append(temp_list)
    label.append(row['hpo_one_hot'])

input_features = np.array(feature)
labels = np.array(label)

input_size = input_features.shape[1]
print("input_size",input_size)
output_size = labels.shape[1]
print("output_size",output_size)
hidden_size = 1024

K = 5
kfold = KFold(n_splits=K, shuffle=True)

true_label = []
predict_label = []
score = []
ave_result = {'aupr':0,'F-max':0}

mask_ma1 = []
for sample_num in range(output_size):
    temp_list = [0.0]*output_size
    temp_list[sample_num] = 1.0
    mask_ma1.append(temp_list*feature_num)
mask_ma1 = torch.tensor(mask_ma1, dtype=torch.float32)

for fold, (train_index, val_index) in enumerate(kfold.split(input_features)):

    train_features, val_features = input_features[train_index], input_features[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]

    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    val_features = torch.tensor(val_features, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.float32).to(device)
    mask_ma1 = mask_ma1.to(device)
    model = Fuse_Model(output_size,mask_ma1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    num_epochs = 3500
    batch_size = 1024
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, train_features.size(0), batch_size):
            inputs = train_features[i:i+batch_size]
            target = train_labels[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)


            loss = criterion(outputs, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        if epoch%10 == 0:
            print(f"Fold: {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    val_inputs = val_features
    val_predictions = model(val_inputs)
    val_loss = criterion(val_predictions, val_labels)
    Y_test = val_labels.detach().cpu()
    y_score = val_predictions.detach().cpu()
    perf_cc = get_results(Y_test, y_score)
    score.append(perf_cc)
    print(perf_cc)
    print(f"Fold: {fold+1}, Validation Loss: {val_loss}")

for k_score in score:
    for a in k_score['all']:
        ave_result[a] += k_score['all'][a]
for b in ave_result:
    ave_result[b] = ave_result[b]/K
print("交叉验证最终结果是：",ave_result)
df = pd.DataFrame(ave_result,index=[0])
df = pd.DataFrame(df.values.T, columns=df.index, index=df.columns)
df = df.transpose()
df.to_csv("fam_dom_vgae.csv", encoding='utf_8_sig')


