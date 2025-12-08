"""
包含了一些训练的工具函数
"""

import torch
from torch import nn
from d2l import torch as d2l

def accuracy(y_hat, y):
    """返回正确预测到数量

    Args:
        y_hat (tensor): 预测概率分布矩阵
        y (tensor): 正确类别的索引值向量

    Returns:
        (float): 正确预测数量
    """
    # one hot 矩阵，行是同一个物体的预测结果（只有一个1），第一维度大小决定了 bs
    if (len(y_hat.shape) > 1) and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)    # 取出最大的数值所在的列（第1维）的索引
    cmp = (y_hat.type(y.dtype) == y)    # 得到一个 bool 类型形状相同的tensor
    return float(cmp.type(y.dtype).sum()) # 把 bool 转化为 int 然后求和 最后转 float

def evaluate_accuracy(net, data_iter):
    """计算预测精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 改成评估模式
    metric = d2l.Accumulator(2) # 两个数值，正确预测数量，预测总量
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())  # y.numel() 返回 tensor 元素数量
    return metric[0] / metric[1]

def evaluate_loss(net, data_iter, loss):
    """ d2l 内部有这个函数, 用于累加所有的 loss"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel()) # sum 张量求和, numel 元素个数总和
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()  # loss 求和并进行梯度运算
            updater(X.shape[0])
        
        # loss 取标量去除梯度
        metric.add(float(l.sum().item()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    # 这是一个数据可视化的函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) # 训练一个 epoch
        test_acc = evaluate_accuracy(net, test_iter)    # 测试评估训练的结果
        animator.add(epoch + 1, train_metrics + (test_acc,))    # 绘制结果
        print(f"Epoch [{epoch + 1}], Train loss {train_metrics[0]:.3f}, Test acc {train_metrics[1]:.3f}")