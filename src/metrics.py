import numpy as np

def rr(gt, pred, k=15):
    gt = gt.split(',')
    pred = pred.split(',')[:k] if pred is not np.nan else ''
    R = 0
    for j, p in enumerate(pred):
        if p in gt:
            R = j +1
            break
    return 1 / R if R else 0


def mrr(df, col1='gold', col2='pred', k=15):
    return np.mean([rr(gt, pred, k=k) for gt, pred in zip(df[col1], df[col2])])

def ap(gt, pred, k=15):
    gt = gt.split(',')
    pred = pred.split(',')[:k] if pred is not np.nan else ''
    return np.sum([len(set(pred[:i+1]).intersection(gt)) / len(pred[:i+1]) for i in range(len(pred)) if pred[i] in gt]) / len(gt)

def map(df, col1='gold', col2='pred', k=15):
    return np.sum([ap(gt, y_pred, k=k) for gt, y_pred in zip(df[col1], df[col2])]) / len(df)