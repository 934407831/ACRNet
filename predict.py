from model.odas import ODAS
from model.loss import Loss
from tool.dataset import data_deal
from torch_lib import traverse
from tool.metrics import accuracy, precision, recall, HM, DSC, IOU, f1
import torch
import os


model = ODAS()
model.load_state_dict(torch.load('model/saved_model/model.pt'))
train_dataset, test_dataset, neg_rate, _ = data_deal(batch_size=1, val_ratio=0, train_ratio=0, shuffle=False)
loss = Loss(weight=torch.tensor([1 - neg_rate, neg_rate]))
dataset = test_dataset
img_path = os.listdir('dataset/data_x')


def callback(data):
    y_pred = data['y_pred']  # 模型预测
    # y_pred = (y_pred[0] + y_pred[1] + y_pred[2] + y_pred[3]) / 4  这里需要改一下，模型的输出不一样
    print(y_pred.shape)
    print(data['metrics'])  # 计算的metrics结果
    print(img_path[data['step']])  # 通过当前step索引得到文件路径


traverse(model, dataset, callbacks=[callback], metrics=[(loss, 'loss'), accuracy, precision, recall, HM, DSC, IOU, f1], console_print=False)
