"""
暂退法, Dropout from scratch & concise
"""

import torch
from torch import nn
from d2l import torch as d2l
from tools import *

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        # 全部元素都被丢弃
        return torch.zeros_like(X)
    elif dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()  # torch.rand 生成随机分布的[0, 1)
    return mask * X / (1.0 - dropout)

class NetScratch(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(NetScratch, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练时候才使用 dropout
        if self.training == True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == "__main__":
    FROM_SCRATCH = False
    # 这里用的是 Fasion-MNIST
    dropout1, dropout2 = 0.2, 0.5
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    if FROM_SCRATCH:
        net = NetScratch(num_inputs, num_outputs, num_hiddens1, num_hiddens2, True)
    else:
        net = nn.Sequential(nn.Flatten(),
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Dropout(dropout1), # 用 API 直接加一个 dropout 层就行了
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(dropout2),
                nn.Linear(256, 10))
    
    net.apply(init_weight) # 参数初始化


    # 训练参数
    num_epochs, lr, batch_size = 10, 0.5, 256

    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


    d2l.plt.show()
