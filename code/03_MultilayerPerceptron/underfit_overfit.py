"""
欠拟合和过拟合
"""
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from tools import *

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1] # 拿到最后一行的输入
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


if __name__ == "__main__":
    
    max_degree = 20 # 最大阶数
    n_train, n_test = 100, 100 # 数据集大小

    # 这是真实的权重, 也是学习的目标
    true_w = np.zeros(max_degree) # 分配空间
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 这是真实的数据

    # 生成 (n_train + n_test, 1) 形状的二维数组, N(0, 1) 正态分布
    features = np.random.normal(size=(n_train + n_test, 1)) # feature 是自变量, 模型输入的一维特征
    np.random.shuffle(features)
    # 生成矩阵运算结果
    exponents = np.arange(max_degree)          # shape = (20,)
    exponents = exponents.reshape(1, -1)       # shape = (1, 20)
    poly_features = np.power(features, exponents)   # shape = (200, 20) 广播运算
    for i in range(max_degree):
        # 为了防止过大的梯度,一般在次方分母添加阶乘, 列方向的下标代表
        poly_features[:, i] /= math.gamma(i + 1) # gamma(n)=(n-1)! 

    # labels的维度: (n_train + n_test, 1) label 是给模型用于训练的数据
    labels = np.dot(poly_features, true_w) # 完全没有误差的数据
    labels += np.random.normal(scale=0.1, size=labels.shape) # 这个是噪声

    # 全部都转化为 tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=
        torch.float32) for x in [true_w, features, poly_features, labels]]
    
    # 3维度, 正常拟合
    train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
    # 1维度, 欠拟合
    train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
    # 20维度, 过拟合
    train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
    d2l.plt.show()