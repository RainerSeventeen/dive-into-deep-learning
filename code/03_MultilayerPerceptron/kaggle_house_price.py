"""kaggle 房价预测"""

import torch
import pandas as pd
from torch import nn
from d2l import torch as d2l

def preprocess(train_path, test_path):
    # 这里已经提前下载好了数据集

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print(train_data.shape)
    print(test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1,]])

    # 注意到有些数据是缺失的, 所以我们需要前处理
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))
    numeric_feature = all_features.dtypes[all_features.dtypes != 'object'].index    # 只保留数值类型的列索引
    all_features[numeric_feature] = all_features[numeric_feature].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    # 所有的数据都已经被归一化, 均值为0
    all_features[numeric_feature] = all_features[numeric_feature].fillna(0)

    print(all_features.shape)
    all_features = pd.get_dummies(all_features, dummy_na=True) # 把 na 视为一个有效的特征
    print(all_features.shape)

    n_train = train_data.shape[0]
    # 转化为 tensor 
    all_features = all_features.astype('float32')   # 全部转化为 float32
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    return train_features, train_labels, test_features, test_data

class MyNet(nn.Module):
    """单一的线性网络"""
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear1(x)

def log_rmse(net, features, labels):
    """对数 rmse loss"""
    # clamp 把数值限定在 [1, inf] 之间
    clipped_peds =  torch.clamp(net(features), 1, float("inf"))
    rmse = torch.sqrt(loss(torch.log(clipped_peds),
                           torch.log(labels)))
    return rmse.item()
    

def train(net, train_features, train_labels, val_features, val_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """训练入口函数"""

    train_ls, val_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 利用 Adam 进行优化
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        
        train_ls.append(log_rmse(net, train_features, train_labels))
        if val_ls is not None: 
            val_ls.append(log_rmse(net, val_features, val_labels))

    return train_ls, val_ls

def get_k_fold_data(k, i, X, y):
    """K 折交叉验证实现, 没有打乱顺序"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size) # 创建对应折的 索引序列
        X_part, y_part = X[idx, :], y[idx]  # 拿出对应的实际数据
        if j == i:  # 第 i 折放进 val
            X_val, y_val = X_part, y_part
        elif X_train is None:   # 第一次放进来的时候是空的, 直接赋值
            X_train, y_train = X_part, y_part
        else:   # 用 cat 附加在末尾
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_val, y_val

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, val_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # train_features.shape[0] 是 N, 样本数量
        net = MyNet(train_features.shape[1])
        # *data 自动解开元组全部依次输入
        train_ls, val_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1] # 只拿最后一个 epoch
        val_l_sum += val_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, val_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            prediction(net)
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(val_ls[-1]):f}')
        return train_l_sum / k, val_l_sum / k

def prediction(net):
    """预测结果并把结果格式化为可提交模式"""
    preds = net(test_features).detach().numpy()
    # 重新格式化
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train_path = "/Users/rainer/Desktop/dive-into-deep-learning/dataset/kaggle_house_price/train.csv"
    test_path = "/Users/rainer/Desktop/dive-into-deep-learning/dataset/kaggle_house_price/test.csv"
    train_features, train_labels, test_features, test_data = preprocess(train_path, test_path)

    loss = nn.MSELoss()
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                            weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
        f'平均验证log rmse: {float(valid_l):f}')

    d2l.plt.show()

