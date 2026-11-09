# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from torch import tensor, float32
# from sklearn.preprocessing import StandardScaler

# # 自定义 Bike 数据集类
# class BikeDataset(Dataset):
#     def __init__(self, file_path):
#         # 加载数据
#         data = pd.read_csv(file_path)
#         self.features = tensor(data.iloc[:, :-1].values, dtype=float32)  # 前 n-1 列为特征
#         self.labels = tensor(data.iloc[:, -1].values, dtype=float32)     # 最后一列为标签（Rented Bike Count）

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# def Bike(data_path,balance = True):
#     data_path = "/root/autodl-tmp/EVA/data/bike"
#     # 数据文件路径
#     train_path = f"{data_path}/train.csv"
#     valid_path = f"{data_path}/valid.csv"
#     test_path = f"{data_path}/test.csv"

#     # 加载数据集
#     train_dataset = BikeDataset(train_path)
#     valid_dataset = BikeDataset(valid_path)
#     test_dataset = BikeDataset(test_path)

#     # 获取数据维度
#     channel = 1  # Bike 数据是 tabular 数据，假设单通道
#     im_size = (train_dataset.features.shape[1],)  # 输入的特征维度
#     num_classes = 1  # 回归任务，输出为 1 个值（Rented Bike Count）
#     class_names = ["Rented Bike Count"]  # 类别名称（回归任务通常不需要类别名称）

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
from torch.utils.data import Dataset, DataLoader
from torch import tensor, float32
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class BikeDataset(Dataset):
    def __init__(self, features, labels, feature_scaler=None, label_scaler=None):
        # 特征标准化
        if feature_scaler is not None:
            self.features = tensor(feature_scaler.transform(features), dtype=float32)
        else:
            self.features = tensor(features, dtype=float32)
        
        # 标签标准化
        if label_scaler is not None:
            # 保存原始标签用于后续逆变换
            self.raw_labels = labels.copy()
            self.labels = tensor(label_scaler.transform(labels.reshape(-1, 1)), dtype=float32).squeeze()
        else:
            self.labels = tensor(labels, dtype=float32)
            self.raw_labels = labels.copy()
            
        # 保存scaler引用
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def Bike(data_path, balance=True):
    data_path = "/data/dengqiyan/EVA/data/bike"
    
    # 加载训练数据
    train_data = pd.read_csv(f"{data_path}/train.csv")
    test_data = pd.read_csv(f"{data_path}/test.csv")

    # 拆分特征和标签
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values.reshape(-1, 1)  # 保持二维结构
    
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

    # 创建标准化器
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    # 训练集标准化
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = label_scaler.fit_transform(y_train)

    # 验证集拆分和标准化（使用训练集的scaler）
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )

    # 测试集标准化（使用训练集的scaler）
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = label_scaler.transform(y_test)

    # 创建数据集（不再需要后期处理）
    train_dataset = BikeDataset(
        X_train_split, y_train_split, 
        feature_scaler=None,  # 已经提前处理
        label_scaler=None
    )
    
    valid_dataset = BikeDataset(
        X_val_split, y_val_split,
        feature_scaler=None,
        label_scaler=None
    )

    test_dataset = BikeDataset(
        X_test_scaled, y_test_scaled,
        feature_scaler=None,
        label_scaler=None
    )

    # 保存scaler用于后续逆变换
    train_dataset.feature_scaler = feature_scaler
    train_dataset.label_scaler = label_scaler
    valid_dataset.label_scaler = label_scaler
    test_dataset.label_scaler = label_scaler

    # 数据维度信息
    channel = 1
    im_size = (X_train.shape[1],)
    num_classes = 1
    class_names = ["count"]

    return channel, im_size, num_classes, class_names, feature_scaler.mean_, feature_scaler.scale_, train_dataset, valid_dataset, test_dataset
