import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import json
from tensorflow.keras.layers import Dense, Layer
import tensorflow.keras.backend as K
from evaluation import get_results

class MaskedLinear(Layer):
    def __init__(self, units, mask, **kwargs):
        super(MaskedLinear, self).__init__(**kwargs)
        self.units = units
        self.mask = mask

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        masked_weight = self.w * self.mask
        return K.dot(inputs, masked_weight) + self.b

class Model(tf.keras.Model):
    def __init__(self, input_size, output_size, mask_ma1):
        super(Model, self).__init__()
        self.weight_layer1 = MaskedLinear(output_size, mask_ma1)
        # self.weight_layer2 = MaskedLinear(output_size, mask_ma2)
        self.outlayer = Dense(output_size, activation='sigmoid')

    def call(self, inputs):
        x = self.weight_layer1(inputs)
        x = tf.nn.leaky_relu(x)
        x = self.outlayer(x)
        return x

def Evaluation_result(predict_result, hpo_list, feature_name):
    Y_test = hpo_list
    y_score = predict_result
    perf_cc = get_results(Y_test, y_score)
    print(feature_name, ":", perf_cc['all'])

# 读取数据
uniprot = pd.read_pickle("../../data/features_20211010.pkl")
print(uniprot.columns)

# 数据处理
feature = []
hpo_list = []

for i, row in uniprot.iterrows():
    temp_list = []
    temp_list.extend(row['vgae_result'])
    # temp_list.extend(row['seq_result'])
    feature.append(temp_list)
    hpo_list.append(row['hpo_one_hot'])

input_features = np.array(feature)
labels = np.array(hpo_list)

input_size = input_features.shape[1]
output_size = labels.shape[1]
print("Input size:", input_size, "Output size:", output_size)

mask_ma1 = []
feature_num = 1
for sample_num in range(output_size):
    temp_list = [0.0]*output_size
    temp_list[sample_num] = 1.0
    mask_ma1.append(temp_list*feature_num)


# 设置 K 折交叉验证
K = 5
kfold = KFold(n_splits=K, shuffle=True)
predictions = []
# 模型训练和评估
for fold, (train_index, val_index) in enumerate(kfold.split(input_features)):
    # 划分训练集和验证集
    train_features, val_features = input_features[train_index], input_features[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]

    # 创建模型实例
    model = Model(input_size, output_size, mask_ma1)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # 训练模型
    model.fit(train_features, train_labels, epochs=20, batch_size=1024)

    # 在验证集上评估模型
    val_loss = model.evaluate(val_features, val_labels)
    print(f"Fold {fold + 1}, Validation Loss: {val_loss}")

    # 对验证集进行预测
    val_predictions = model.predict(val_features)
    predictions = predictions + val_predictions.tolist()

Evaluation_result(predictions, hpo_list, "seq_result")