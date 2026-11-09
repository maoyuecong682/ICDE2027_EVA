# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from torch import tensor, float32, long
# from sklearn.preprocessing import StandardScaler

# # 自定义 HR 数据集类
# class HRDataset(Dataset):
#     def __init__(self, file_path):
#         # 加载数据
#         data = pd.read_csv(file_path)
#         self.features = tensor(data.iloc[:, :-1].values, dtype=float32)  # 前 13 列为属性
#         self.labels = tensor(data.iloc[:, -1].values, dtype=long)        # 最后一列为标签

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# def HR(data_path, balance=True):
#     data_path = "/data/dengqiyan/EVA/data/HR"
#     # 数据文件路径
#     train_path = f"{data_path}/train.csv"
#     valid_path = f"{data_path}/valid.csv"
#     test_path = f"{data_path}/test.csv"

#     # 加载数据集
#     train_dataset = HRDataset(train_path)
#     valid_dataset = HRDataset(valid_path)
#     test_dataset = HRDataset(test_path)

#     # 获取数据维度
#     channel = 1  # HR 数据是 tabular 数据，假设单通道
#     im_size = (13,)  # 输入的特征维度
#     num_classes = len(train_dataset.labels.unique())  # 根据训练集的标签推断类别数
#     class_names = [str(i) for i in range(num_classes)]  # 类别名称假设为字符串

#     # 数据标准化
#     scaler = StandardScaler()
#     train_dataset.features = tensor(
#         scaler.fit_transform(train_dataset.features), dtype=float32
#     )
#     valid_dataset.features = tensor(
#         scaler.transform(valid_dataset.features), dtype=float32
#     )
#     test_dataset.features = tensor(
#         scaler.transform(test_dataset.features), dtype=float32
#     )
#     mean = scaler.mean_
#     std = scaler.scale_

#     # 返回必要的信息和数据集
#     return channel, im_size, num_classes, class_names, mean, std, train_dataset, valid_dataset, test_dataset

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch import tensor, float32, long
from sklearn.preprocessing import StandardScaler

# 自定义 HR 数据集类
class HRDataset(Dataset):
    def __init__(self, file_path):
        # 加载数据
        data = pd.read_csv(file_path)
        self.features = tensor(data.iloc[:, :-1].values, dtype=float32)  # 前 13 列为属性
        self.labels = tensor(data.iloc[:, -1].values, dtype=long)        # 最后一列为标签

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def HR(data_path, balance=True):
    data_path = "/data/dengqiyan/EVA/data/HR"
    # 数据文件路径
    train_path = f"{data_path}/train.csv"
    valid_path = f"{data_path}/valid.csv"
    test_path = f"{data_path}/test.csv"

    # 加载数据集
    train_dataset = HRDataset(train_path)
    test_dataset = HRDataset(test_path)

    # 计算验证集的大小为 20% 的训练集
    valid_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - valid_size

    # 使用 random_split 将训练集分割成训练集和验证集
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    # 获取数据维度
    channel = 1  # HR 数据是 tabular 数据，假设单通道
    im_size = (13,)  # 输入的特征维度
    num_classes = len(train_dataset.dataset.labels.unique())  # 根据训练集的标签推断类别数
    class_names = [str(i) for i in range(num_classes)]  # 类别名称假设为字符串

    # 数据标准化
    scaler = StandardScaler()
    train_dataset.dataset.features = tensor(
        scaler.fit_transform(train_dataset.dataset.features), dtype=float32
    )
    valid_dataset.dataset.features = tensor(
        scaler.transform(valid_dataset.dataset.features), dtype=float32
    )
    test_dataset.features = tensor(
        scaler.transform(test_dataset.features), dtype=float32
    )
    mean = scaler.mean_
    std = scaler.scale_

    # 返回必要的信息和数据集
    return channel, im_size, num_classes, class_names, mean, std, train_dataset, valid_dataset, test_dataset
