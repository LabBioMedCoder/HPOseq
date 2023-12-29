import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, normalize
import scipy.sparse as sp
import json
import tensorflow as tf
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from gcnModel import GCNModelAE, GCNModelVAE, VAE
from optimizer import OptimizerAE, OptimizerVAE
import pickle as pkl
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from evaluation import get_results
import re
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

def clean_amino_acid_sequence(sequence):
    # 定义 20 种常见氨基酸
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

    # 移除不在 20 种常见氨基酸中的字符
    cleaned_sequence = ''.join([aa for aa in sequence if aa in valid_amino_acids])

    return cleaned_sequence

def Evaluation_result(predict_result, hpo_list, feature_name):
    Y_test = hpo_list
    y_score = predict_result
    perf_cc = get_results(Y_test, y_score)
    print(feature_name,":",perf_cc['all'])

class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, hidden1, hidden2, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              # act=tf.nn.relu,
                                              act=tf.nn.tanh,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=self.hidden1_dim,
                                          output_dim=self.hidden2_dim,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.compat.v1.random_normal([self.n_samples, self.hidden2_dim], dtype=tf.float64) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=self.hidden2_dim,
                                        act=lambda x: x,
                                      logging=self.logging)(self.z)

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

def train_gcn(features, adj_train, model_str="gcn_vae", hidden1=800, hidden2=400):
    # 存储原始邻接矩阵（无对角线条目）
    adj_orig = adj_train
    # 去除对角线
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # 只要其中的非0部分
    adj_orig.eliminate_zeros()
    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_train)
    adj = adj_train

    # 将稀疏矩阵处理成二维向量
    adj_norm = preprocess_graph(adj)
    # Define placeholders
    tf.compat.v1.disable_eager_execution()
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float64),
        'adj': tf.compat.v1.sparse_placeholder(tf.float64),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float64),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero, hidden1, hidden2)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, hidden1, hidden2)
    elif model_str == 'vae':
        model = VAE(placeholders, num_features, num_nodes, features_nonzero, hidden1, hidden2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm,
                              lr=0.01)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.compat.v1.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                     validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm,
                               lr=0.001)

    # Initialize session
    sess = tf.compat.v1.Session()
    # 初始化模型全局参数
    sess.run(tf.compat.v1.global_variables_initializer())
    #添加对角矩阵
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    epochs = 300

    for epoch in range(epochs):

        # Construct feed dictionary，feed_dict的作用是给使用placeholder创建出来的tensor赋值，feed使用一个值临时替换一个op的输出结果
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: 0})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)  # 执行整个定义好的计算图

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]))

    print("Optimization Finished!")

    # return embedding for each protein
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    return emb

def train_nn(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(1024, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(Y_train.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  #loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=80, batch_size=512, verbose=0)

    y_prob = model.predict(X_test)

    return y_prob

def load_ppi_network(net_list, gene_num):

    data = net_list
    adj = np.zeros((gene_num, gene_num))
    # for x in tqdm(adj_data):
    for x in data:
        temp = x.split()
        # check whether score larger than the threshold
        if float(temp[2]) >= 0:
            adj[int(temp[0]), int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T

    return adj

def reshape(features):
    return np.hstack(features).reshape((len(features), len(features[0])))

def VGAE_model_train(seq_list, net_list, hpo_list):

    print("加载相关文件")
    uniprot = pd.read_pickle("../../data/features_20211010.pkl")
    print(uniprot)

    features = []
    index = 1
    for seq in seq_list:
        temp_tuple = (str(index),clean_amino_acid_sequence(seq))
        features.append(temp_tuple)
        index+=1
    CT_encoded_sequences = CTriad(features)
    feature_list = []
    for CT in CT_encoded_sequences[1:]:
        feature_list.append(CT[1:])

    features = sp.csr_matrix(feature_list)

    adj = load_ppi_network(net_list, uniprot.shape[0])
    print("adj",adj[0])
    adj = sp.csr_matrix(adj)
    embeddings = train_gcn(features, adj)
    print(embeddings)
    print(embeddings)
    with open("../../data/network/network.pkl", 'wb') as file:
        pkl.dump(embeddings, file)


    X_hp_data = embeddings
    hp = np.array(hpo_list)
    kf = KFold(n_splits=5)
    predictions = []

    for train_index, val_index in kf.split(X_hp_data):
        X_train, X_val = X_hp_data[train_index], X_hp_data[val_index]
        y_train, y_val = hp[train_index], hp[val_index]

        val_predictions = train_nn(X_train, y_train, X_val, y_val)

        # 将预测结果存储在相应的位置
        predictions = predictions+val_predictions.tolist()

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

with open('../../data/network/blast_network.txt') as f:  # with语句替代try…except…finally…
    net_list = f.readlines()

vgae_predict = VGAE_model_train(seq_list, net_list, hpo_list)

# print(vgae_predict)
Evaluation_result(vgae_predict, hpo_list, "seq_result")

uniprot['vgae_result'] = vgae_predict

uniprot.to_pickle("../../data/features_20211010.pkl")
