from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
from keras.models import load_model
import numpy as np
from sklearn.model_selection import KFold
import torch
from src.model.evaluation import get_results
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Model_comparison(model1, model2, X_val, y_val):
    model1_result = model1.predict(X_val)
    model2_result = model2.predict(X_val)
    model1_aupr, model1_fmax = Evaluation_result(model1_result, y_val)
    model2_aupr, model2_fmax = Evaluation_result(model2_result, y_val)
    if model1_aupr + model1_fmax < model2_aupr + model2_fmax:
        return model2
    else:
        return model1

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

def reshape(features):
    return np.hstack(features).reshape((len(features), len(features[0])))

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

def CNNModule(input_dim, hidden_size, num_classes):
    model = Sequential([
        Embedding(input_dim, 256, input_length=1998),
        Conv1D(64, 256, padding='valid', activation='relu', strides=1),
        Conv1D(32, 128, padding='valid', activation='relu', strides=1),
        Conv1D(16, 64, padding='valid', activation='relu', strides=1),
        MaxPooling1D(64, 32),
        Flatten(),
        Dense(hidden_size, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='sigmoid')
    ])
    return model

def CNN_model_train(learningrate, batch_size, epochs, hidden_size, seq_list, hpo_list, model_save_path):
    input_dim = 8001
    feature_list = encode_protein_sequences_no_itertools(seq_list)
    X = np.array(feature_list)
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
        model = CNNModule(input_dim, hidden_size, num_classes)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        saved_model = load_model(model_save_path)
        bast_model = Model_comparison(model, saved_model, X_val, y_val)
        bast_model.save(model_save_path)
        val_predictions = bast_model.predict(val_dataset)
        predictions = predictions + val_predictions.tolist()

    return predictions

uniprot = pd.read_pickle('data/features_20211010.pkl')

hpo_list = []
seq_list = []
for i, row in uniprot.iterrows():
    hpo_label = row['hpo_one_hot']
    seq = row['seq']
    hpo_list.append(hpo_label)
    seq_list.append(seq)
intra_predict = CNN_model_train(0.0001, 1024, 80, 2048, seq_list, hpo_list,
                                'data/best_model/best_seq_model.h5')
uniprot['intra_result'] = intra_predict
uniprot.to_pickle("../../data/features_20211010.pkl")
