import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score


def calculate_accuracy(y_test, y_score):
    y_score_max = np.argmax(y_score, axis=1)
    cnt = 0
    for i in range(y_score.shape[0]):
        if y_test[i, y_score_max[i]] == 1:
            cnt += 1

    return float(cnt) / y_score.shape[0]


def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
    return f_max

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
    return f, acc, precision, recall


def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()
    perf['aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha * [1])
    perf['F-max'] = calculate_fmax(y_score, y_test)

    return perf


def get_results(Y_test, y_score):
    perf = defaultdict(dict)
    Y_test = np.array(Y_test, dtype=np.float64)
    y_score = np.array(y_score, dtype=np.float64)

    perf['all'] = evaluate_performance(Y_test, y_score)

    return perf



