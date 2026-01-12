"""
卷积神经网络的实现
"""

import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    # 这是互相关运算, 是倒序的卷积
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j:j + w] * K).sum()
    return Y

def corr2d_v2(X, K, padding, stride):
    """
    2D 互相关（cross-correlation），支持 padding 和 stride。

    假设：
    - padding 和 stride 在 H/W 两个方向相同, padding 指的是所有方向上的填充值(1代表上下左右都填充1次)
    - K 是正方形卷积核 (k, k)
    - X 是二维张量 (H, W)
    """
    # 取核大小（正方形核）
    k, _ = K.shape

    # 输入尺寸 (H, W)
    H, W = X.shape

    # 进行 padding：上下左右各 padding
    if padding > 0:
        X_pad = torch.zeros((H + 2 * padding, W + 2 * padding), dtype=X.dtype, device=X.device)
        X_pad[padding:padding + H, padding:padding + W] = X
    else:
        X_pad = X

    H_pad, W_pad = X_pad.shapeß

    # 输出尺寸：((H + 2p - k) // s + 1, (W + 2p - k) // s + 1)
    out_h = (H_pad - k) // stride + 1
    out_w = (W_pad - k) // stride + 1

    Y = torch.zeros((out_h, out_w), dtype=X.dtype, device=X.device)

    # 按 stride 滑动窗口做互相关
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            Y[i, j] = (X_pad[h_start:h_start + k, w_start:w_start + k] * K).sum()

    return Y

def corr2d_mimo(X, K):
    """多输入多输出通道的 2D 互相关。

    参数
    - X: (C_in, H, W)
    - K: (C_out, C_in, kH, kW)

    返回
    - Y: (C_out, H_out, W_out)
    """
    c_in, H, W = X.shape
    c_out, c_in_k, kH, kW = K.shape

    # 用一次 corr2d 计算输出空间尺寸（H_out, W_out）
    Y0 = corr2d(X[0], K[0, 0])
    H_out, W_out = Y0.shape

    Y = torch.zeros((c_out, H_out, W_out), dtype=X.dtype, device=X.device)

    # 对每个输出通道：累加所有输入通道的互相关
    for o in range(c_out):
        for i in range(c_in):
            Y[o] += corr2d(X[i], K[o, i])

    return Y

def corr2d_mimo_1x1(X, K):
    """
    1x1 卷积, 使用全连接层实现
    """
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w)) # 按照通道数拉直
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))   # 恢复空间信息


class Conv2D(nn.Module):
    def __init__(self, kernal_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernal_size))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
if __name__ == "__main__":
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    X = X.reshape((1, 1, 6, 8)) # batchsize, channel, H, W
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= 1e-2 * conv2d.weight.grad
        print(f"batch {i + 1}, loss {l.sum():.3f}")