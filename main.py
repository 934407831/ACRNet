import torch.cuda
import numpy as np
from model.odas import ODAS
from torchsummary import summary
from tool import device
from tool.dataset import data_deal
from torch_lib import fit, evaluate
from model.loss import Loss
from tool.metrics import precision, recall, accuracy, f1, DSC, HM
from tool.callback import save_log
import time

train_dataset, val_dataset, test_dataset, neg_rate, _ = data_deal(batch_size=2, train_ratio=0.6, val_ratio=0.3)
print(torch.cuda.is_available())
model = ODAS().to(device)
# summary(model, (1, 512, 512))
# print(model)


def call(data):
    mode = data['model']
    for name, parms in mode.named_parameters():
        if name == 'fuse_layer.bias':
            print('-->name:', name, '-->grad_requirs:', parms, ' -->grad_value:', parms.grad)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# lambda1 = lambda epoch: epoch // 10
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=-1)
loss = Loss(weight=torch.tensor([1-neg_rate, neg_rate]).to(device))
fit(model=model, train_dataset=train_dataset, val_dataset=val_dataset, epochs=100, loss_func=loss, metrics=[accuracy, precision, recall, f1, DSC, HM], epoch_callbacks=[save_log], optimizer=optimizer)
# eva = evaluate(model=model, dataset=test_dataset, metrics=[accuracy, precision, recall, f1, DSC, HM], loss_func=loss)
# print(eva)
# with open('log/test_result.txt', 'w') as f:
#     time = time.asctime()
#     str1 = eva
#     f.write(str1)
