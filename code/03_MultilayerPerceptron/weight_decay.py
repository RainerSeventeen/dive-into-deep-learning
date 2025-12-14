"""
权重衰退的实现, 通过高阶数据和过拟合实现
"""

import torch
from torch import nn
from d2l import torch as d2l
from tools import *

def init_params():
    w = torch.normal(0 , 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    """L2 范数惩罚, 可以参见 md 部分公式"""
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # L2 范数惩罚, 系数lambd
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(
                epoch + 1,(evaluate_loss(net, train_iter, loss), 
                           evaluate_loss(net, test_iter, loss)))

def train_concise(wd):
    """简洁实现"""
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    with torch.no_grad():
        for param in net.parameters():
            param.normal_()    # 随机正态分布初始化

    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 这个表示对 weight 做 weight_decay 但是 bias 不做
    trainer = torch.optim.SGD([
        {"params":net[0].weight, 'weight_decay': wd},
        {"params":net[0].bias}],lr=lr)
    
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()

        if (epoch + 1) % 5 == 0:
            animator.add(
                epoch + 1,(evaluate_loss(net, train_iter, loss), 
                           evaluate_loss(net, test_iter, loss)))
    print(f"L2 范数: {net[0].weight.norm().item()}")


if __name__ == "__main__":
    
    # 准备真值数据和测试数据
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

    X_train = torch.normal(0, 1, size=(n_train, num_inputs))
    # y_train: 线性模型 Xw + b，再加一点高斯噪声
    y_train = X_train @ true_w + true_b
    y_train += torch.normal(0, 0.01, size=y_train.shape)
    # 测试数据
    X_test = torch.normal(0, 1, size=(n_test, num_inputs))
    y_test = X_test @ true_w + true_b
    y_test += torch.normal(0, 0.01, size=y_test.shape)

    # dataloader 
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # is_train=False 对应不打乱
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型初始化
    train(lambd=0)
    train(lambd=3)
    train_concise(wd=3)

    d2l.plt.show()
