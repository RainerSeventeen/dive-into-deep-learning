"""
多层感知机实现逻辑 from scratch & concise
"""

import torch
from torch import nn
from d2l import torch as d2l
from tools import train_ch3


def relu(x):
    """ReLU 激活函数"""
    a = torch.zeros_like(x)
    return torch.max(a, x)

def mlp_net(X):
    X = X.reshape(-1, num_input)
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

if __name__ == "__main__":
    FROM_SCRATCH = True

    # 加载数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_input, num_output = 28 * 28, 10
    num_hidden = 256 # 隐藏层大小
    net = None

    if FROM_SCRATCH:
        # 如果这里参数设置为 0 会怎么样呢 ?
        W1 = nn.Parameter(torch.randn(num_input, num_hidden, requires_grad=True))
        b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))

        W2 = nn.Parameter(torch.randn(num_hidden, num_output, requires_grad=True))
        b2 = nn.Parameter(torch.zeros(num_output, requires_grad=True))
        params = [W1, b1, W2, b2]
        net = mlp_net
    else:
        # 简洁实现
        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output),
        )
        params = net.parameters()
    # 下面的训练方式和 softmax 完全相同
    num_epoch = 10
    lr = 0.1
    updator = torch.optim.SGD(params, lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')    # 自动包含了 softmax 功能
    train_ch3(net, train_iter, test_iter, loss, num_epoch, updator)
    d2l.plt.show()

