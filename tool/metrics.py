import torch
import numpy as np


def accuracy(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    return np.mean(stage == y_true)


def precision(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    tp, fp, _, _ = separate_count(stage, y_true)
    return divide(tp, tp + fp)


def recall(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    tp, _, fn, _ = separate_count(stage, y_true)
    return divide(tp, tp + fn)


def f1(y_pred, y_true):
    pre = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return divide(2 * pre * rec, pre + rec)


def DSC(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    tp, fp, fn, _ = separate_count(stage, y_true)
    return divide(2 * tp, fn + 2 * tp + fp)


def HM(y_pred, y_true):
    stage, y_true = parse(y_pred, y_true)
    tp, fp, fn, _ = separate_count(stage, y_true)
    union = tp + fp + fn
    return divide(union - tp, union)


def parse(y_pred, y_true):
    stage = y_pred
    # stage = y_pred[0]
    stage = torch.argmax(stage, dim=1)
    # 转换为np数组用于计算
    stage = np.array(stage.cpu())
    y_true = np.array(y_true.cpu())
    return stage, y_true


def separate_count(stage, y_true):
    tp = np.sum((stage == y_true) & (y_true == 1))
    fp = np.sum((stage != y_true) & (y_true == 0))
    fn = np.sum((stage != y_true) & (y_true == 1))
    tn = np.sum((stage == y_true) & (y_true == 0))
    return tp, fp, fn, tn


def divide(dividend, divisor):
    return dividend / divisor if divisor != 0 else 0.0
