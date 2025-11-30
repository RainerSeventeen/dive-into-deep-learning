"""
使用 pytorch API 来实现 softmax
"""

import torch
from torch import nn
from d2l import torch as d2l
from softmax_from_scratch import train_ch3


def init_weight(m):
    """参数初始化"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == "__main__":
    batch_size = 256
    # 使用 d2l 的 API
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    net = nn.Sequential(
        nn.Flatten(),   # 在线性层前展开
        nn.Linear(28 * 28, 10)  # pytorch 会自动在内部执行 softmax 的计算
    )
    net.apply(init_weight) # 递归按层执行这个函数，函数输入固定为 module，返回 None
    loss = nn.CrossEntropyLoss(reduction='none') # 这里不执行平均值，因为在 train_ch3返回的时候会自动求平均
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10

    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    d2l.plt.show() # 训练结果展示图片