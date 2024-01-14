import scipy.sparse as sp
from src.model.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from src.model.gcnModel import GCNModelAE, GCNModelVAE, VAE
from src.model.optimizer import OptimizerAE, OptimizerVAE
from keras.models import Sequential
from keras.layers import Dense, Dropout
import re
from src.model.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
from src.model.evaluation import get_results
import warnings
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    cleaned_sequence = ''.join([aa for aa in sequence if aa in valid_amino_acids])

    return cleaned_sequence


def Evaluation_result(predict_result, hpo_list, feature_name=''):
    Y_test = hpo_list
    y_score = predict_result
    perf_cc = get_results(Y_test, y_score)
    if feature_name != '':
        print(feature_name, ":", perf_cc['all'])
    return perf_cc['all']['aupr'], perf_cc['all']['F-max']


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

        self.z = self.z_mean + tf.compat.v1.random_normal([self.n_samples, self.hidden2_dim],
                                                          dtype=tf.float64) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=self.hidden2_dim,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.z)


def Model_comparison(model1, model2, X_val, y_val):
    model1_result = model1.predict(X_val)
    model2_result = model2.predict(X_val)
    model1_aupr, model1_fmax = Evaluation_result(model1_result, y_val)
    model2_aupr, model2_fmax = Evaluation_result(model2_result, y_val)
    if model1_aupr + model1_fmax < model2_aupr + model2_fmax:
        return model2
    else:
        return model1


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + gap + 1 < len(sequence) and i + 2 * gap + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + gap + 1]] + '.' + AADict[
                    sequence[i + 2 * gap + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def CTriad(fastas, gap=0, **kw):
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

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

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


def train_gcn(features, adj_train, model_str="gcn_vae", hidden1=2048, hidden2=1024):
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_train)
    adj = adj_train
    
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
    sess.run(tf.compat.v1.global_variables_initializer())
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    epochs = 50

    for epoch in range(epochs):

        # Construct feed dictionary，feed_dict的作用是给使用placeholder创建出来的tensor赋值，feed使用一个值临时替换一个op的输出结果
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: 0})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)  # 执行整个定义好的计算图

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]))

    # return embedding for each protein
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    return emb


def NNModule(hidden_size, num_classes):
    model = Sequential([
        Dense(hidden_size, activation='relu'),
        Dropout(0.3),
        Dense(hidden_size, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='sigmoid')
    ])
    return model


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


def NN_model_train(learningrate, batch_size, epochs, hidden_size, seq_emb_list, hpo_list, model_save_path):
    X = seq_emb_list
    Y = np.array(hpo_list)
    kf = KFold(n_splits=5)
    predictions = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        num_classes = y_train.shape[1]
        optimizer = Adam(learning_rate=learningrate)
        model = NNModule(hidden_size, num_classes)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        saved_model = load_model(model_save_path)
        bast_model = Model_comparison(model, saved_model, X_val, y_val)
        bast_model.save(model_save_path)
        val_predictions = bast_model.predict(val_dataset)
        predictions = predictions + val_predictions.tolist()
    return predictions


def VGAE_model_train(seq_list, net_list):
    features = []
    index = 1
    for seq in seq_list:
        temp_tuple = (str(index), clean_amino_acid_sequence(seq))
        features.append(temp_tuple)
        index += 1
    CT_encoded_sequences = CTriad(features)
    feature_list = []
    for CT in CT_encoded_sequences[1:]:
        feature_list.append(CT[1:])

    features = sp.csr_matrix(feature_list)

    adj = load_ppi_network(net_list, len(seq_list))
    adj = sp.csr_matrix(adj)
    embeddings = train_gcn(features, adj)

    return embeddings


def encode_protein_sequences_no_itertools(protein_sequences):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    trinucleotide_dict = {}
    idx = 0
    for a1 in amino_acids:
        for a2 in amino_acids:
            for a3 in amino_acids:
                trinucleotide_dict[f'{a1}{a2}{a3}'] = idx
                idx += 1
    trinucleotide_dict['000'] = idx

    def encode_sequence(sequence):
        cleaned_sequence = ''.join([aa for aa in sequence if aa in amino_acids])
        if len(cleaned_sequence) >= 2000:
            cleaned_sequence = cleaned_sequence[:2000]
        else:
            cleaned_sequence += '0' * (2000 - len(cleaned_sequence))
        encoded_sequence = [trinucleotide_dict.get(cleaned_sequence[i:i + 3], trinucleotide_dict['000'])
                            for i in range(2000 - 2)]

        return encoded_sequence
    encoded_sequences = [encode_sequence(seq) for seq in protein_sequences]

    return encoded_sequences

uniprot = pd.read_pickle('data/features_20211010.pkl')
print(uniprot.columns)

hpo_list = []
seq_list = []
for i, row in uniprot.iterrows():
    hpo_label = row['hpo_one_hot']
    seq = row['seq']
    hpo_list.append(hpo_label)
    seq_list.append(seq)

with open('../../data/network/blast_network_0.01.txt') as f:
    net_list = f.readlines()

seq_vgae_emb = VGAE_model_train(seq_list, net_list)

inter_predict = NN_model_train(0.0001, 1024, 80, 2048, seq_vgae_emb, hpo_list,
                               'data/best_model/best_inter_model.h5')
uniprot['inter_result'] = inter_predict
uniprot.to_pickle("../../data/features_20211010.pkl")
