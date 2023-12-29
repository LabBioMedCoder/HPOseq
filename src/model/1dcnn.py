import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from evaluation import get_results
import pandas as pd


def Evaluation_result(predict_result, hpo_list, feature_name):
    Y_test = hpo_list
    y_score = predict_result
    perf_cc = get_results(Y_test, y_score)
    print(feature_name, ":", perf_cc['all'])


def encode_protein_sequences_no_itertools(protein_sequences):
    # Define the 20 common amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

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


class CNNModule(tf.keras.Model):
    def __init__(self, input_dim, hidden_size, num_classes):
        super(CNNModule, self).__init__()
        self.embedding = Embedding(input_dim, 128)  # 8001 to include '000' and padding index
        self.conv1 = Conv1D(64, 256, activation='relu')
        self.conv2 = Conv1D(32, 128, activation='relu')
        self.conv3 = Conv1D(16, 64, activation='relu')
        self.pool = MaxPooling1D(64)
        self.flatten = Flatten()
        self.fc1 = Dense(2048, activation='relu')
        self.fc2 = Dense(1024, activation='relu')
        self.fc3 = Dense(num_classes, activation='sigmoid')  # Adjust num_classes as needed

    def call(self, x):
        x = self.embedding(x)
        print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def CNN_model_train(learningrate, batch_size, epochs, hidden_size, seq_list, hpo_list):
    # 模型初始化
    input_dim = 8001
    hidden_size = hidden_size
    num_classes = len(hpo_list[0])

    feature_list = encode_protein_sequences_no_itertools(seq_list)

    # print(len(feature_list),len(feature_list[0]))

    X = np.array(feature_list)
    y = np.array(hpo_list)

    kf = KFold(n_splits=5)

    # 创建一个数组来存储所有预测结果
    predictions = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 转换为 TensorFlow 数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        optimizer = Adam(learning_rate=learningrate)
        model = CNNModule(input_dim, hidden_size, num_classes)

        # 编译模型
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # 训练模型
        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size)

        # 对验证集进行预测
        val_predictions = model.predict(val_dataset)
        # 将预测结果存储在相应的位置
        predictions = predictions + val_predictions.tolist()
        # 评估模型
        val_loss, val_acc = model.evaluate(val_dataset)
        print(f"Validation accuracy: {val_acc}, Validation loss: {val_loss}")

    return predictions


uniprot = pd.read_pickle('../../data/features_20211010.pkl')
print(uniprot.columns)

hpo_list = []
seq_list = []
for i, row in uniprot.iterrows():
    hpo_label = row['hpo_one_hot']
    seq = row['seq']
    hpo_list.append(hpo_label)
    seq_list.append(seq)

seq_predict = CNN_model_train(0.0001, 512, 20, 2048, seq_list, hpo_list)  # 80
# print(seq_predict.tolist())
Evaluation_result(seq_predict, hpo_list, "seq_result")
uniprot['seq_result'] = seq_predict
uniprot.to_pickle("../../data/features_20211010.pkl")