"""
从零开始实现 Softmax 线性回归
"""

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


def load_data_fashion_mnist(batch_size, resize=None):
    """
    加载并下载 fasion mnist 数据集
    原始的输入的是 28 * 28
    """
    trans = transforms.ToTensor()

    if resize:
        trans.insert(0, transforms.Resize(resize))
    # 60k
    minist_train = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=True,
        transform=trans, download=True
    )
    # 10k
    minist_test = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=False,
        transform=trans, download=True
    )

    # torch.Size([1, 28, 28])
    # print(minist_train[0][0].shape)

    # 读取数据
    num_worker = 4
    train_iter = data.DataLoader(minist_train, batch_size, shuffle=True,
                                 num_workers=num_worker)
    test_iter = data.DataLoader(minist_test, batch_size, shuffle=True,
                                 num_workers=num_worker)
    
    return train_iter, test_iter

def softmax(X):
    """softmax 运算"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True) # 对矩阵行进行求和
    return X_exp / partition # 使用了广播机制

def cross_entropy(y_hat, y):
    # 拿到真实数值
    num_label = range(len(y_hat))
    poss = y_hat[num_label, y] # 拿到真实数值的对应值
    return -torch.log(poss)

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


class Accumulator:  #@save
    # 这是一个 d2l 内部已经存在的工具 d2l.Accumulator
    """在n个变量上累加"""
    def __init__(self, n):
        # 创建一 n 维度的列表
        self.data = [0.0] * n 

    def add(self, *args):
        # 把所有输入的参数逐个累加
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # 全部清空
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        # 按照下标可以直接获取某个值
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算预测精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 改成评估模式
    metric = Accumulator(2) # 两个数值，正确预测数量，预测总量
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())  # y.numel() 返回 tensor 元素数量
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
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
    # train_loss, train_acc = train_metrics

if __name__ == "__main__":
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)

    # softmax 的输入需要拉直为一个向量
    num_inputs = 28 * 28
    num_outputs = 10     # 数据集有10个类别

    # 定义参数
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs),
                     requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    loss = cross_entropy
    train_epoch = 5
    lr = 0.1

    def net(X):
        """Softmax 回归网络"""
        # 矩阵乘法要归一化长度和宽度
        return softmax(torch.matmul(X.reshape(-1, w.shape[0]), w) + b)

    def updater(batch_size):
        return d2l.sgd([w, b], lr, batch_size)

    train_ch3(net, train_iter, test_iter, loss, train_epoch, updater)
    
    d2l.plt.show() # 训练结果展示图片
