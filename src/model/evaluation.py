import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def calculate_accuracy(y_test, y_score):
    y_score_max = np.argmax(y_score,axis=1)
    cnt = 0
    for i in range(y_score.shape[0]):
        if y_test[i, y_score_max[i]] == 1:
            cnt += 1
    
    return float(cnt)/y_score.shape[0]

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max


def plot_PRCurve(label, score):
    precision, recall, _ = precision_recall_curve(label.ravel(), score.ravel())
    aupr = average_precision_score(label, score, average="micro")
    
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b',
                 **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(aupr))
    plt.show()

def calculate_f1_score(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    threshold = 0.5
    predictions = (preds > threshold).astype(np.int32)
    p0 = (preds < threshold).astype(np.int32)
    tp = np.sum(predictions * labels)
    fp = np.sum(predictions) - tp
    fn = np.sum(labels) - tp
    tn = np.sum(p0) - fn
    sn = tp / (1.0 * np.sum(labels))
    sp = np.sum((predictions ^ 1) * (labels ^ 1))
    sp /= 1.0 * np.sum(labels ^ 1)
    fpr = 1 - sp
    precision = tp / (1.0 * (tp + fp))
    recall = tp / (1.0 * (tp + fn))
    f = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return f,acc,precision,recall
def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()
    
    perf["M-aupr"] = 0.0
    perf["M-auc"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        passlist = []
        num_pos = sum(y_test[:, i])
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            auc = roc_auc_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            perf["M-auc"] += auc
            aupr_list.append(ap)
            num_pos_list.append(num_pos)
    perf["M-aupr"] /= n
    perf["M-auc"] /= n
    # perf['aupr_list'] = aupr_list
    # perf['num_pos_list'] = num_pos_list
    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    # Computes accuracy
    perf['accuracy'] = calculate_accuracy(y_test, y_score)

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    # perf["F1-score"] = f1_score(y_test, y_new_pred, average='macro')
    perf['F1-score'], perf['accuracy'], perf['precision'], perf['recall'] = calculate_f1_score(y_score, y_test)
    perf['F-max'] = calculate_fmax(y_score, y_test)

    return perf

def get_results(Y_test, y_score):
    perf = defaultdict(dict)
    Y_test = np.array(Y_test, dtype=np.float64)
    y_score = np.array(y_score, dtype=np.float64)
    
    # print('Y_test',Y_test)
    # print('y_score',y_score)
    perf['all'] = evaluate_performance(Y_test, y_score)
    
    return perf
    
    
    
    