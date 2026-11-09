import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
import argparse
import logging

args = argparse.Namespace(
    # data_path='/data2/wyh-dataset/image-dataset',
    # data_path='/data2/wyh-dataset/text-dataset/imdb',
    # data_path='/data2/wyh-dataset/text-dataset/News',
    data_path='/data2/wyh-dataset/tabular-dataset',
    dataset='HR',
    model='TextMLP',   
    method='kmeans',
    selection_optimizer='SGD',
    selection_lr=0.01,
    selection_momentum=0.9,
    selection_weight_decay=5e-4,
    selection_nesterov=True,
    selection_batch=128,
    selection_fraction = 0.1,
    step_size=10,
    gamma=0.1,
    print_freq=10,
    gpu=[2],
    epochs=100,
    workers=4,
    random_seed=0,
    pretrain_epochs=3,
    active_fraction_ratio=0.5,
    Is_pretrain = True,
    proxy_model='ProxyMLP',
    proxy_epochs=3,
    dve_epochs=3,
    dve_lr=0.001,
    selection_interval = 10,
    lambda_reg = 0.5,
    balance = True,
    save_log_path = './log',
    balance_target = 0,
    save_best_model_path = './best_model',
    selection_framework = 'interval',
    patience = 100,
)

log_path = "./log/{}/feature_shift.log".format(args.dataset)
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)

# 替换全局 print
global print
print = lambda *args: logging.info(" ".join(map(str, args)))

import time
# 记录开始时间,现实时间
time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Start time:", time_now)


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False  # 确保可以使用更多的 cuDNN 算法


torch.cuda.set_device(0)

import numpy as np
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from mydatasets import adult, imdb, CIFAR10,news,covtype,MNIST,CIFAR100,Number,Mnist2MnistM,HR
import math
from nets import *
import torch.nn as nn
import torch.optim as optim
from nets.nets_utils import EmbeddingRecorder

from utils.debug import *
from tqdm import tqdm
import random
from import_methods import *
from tqdm.contrib import tenumerate

dataset_map = {
    'adult': adult,
    'imdb': imdb,
    'cifar10': CIFAR10,
    'mnist': MNIST,
    'news' : news,
    'covtype': covtype,
    'cifar100': CIFAR100,
    'number': Number,
    'mnist2mnistm': Mnist2MnistM,
    'HR': HR,
}



# 检查 dataset 是否在映射中
if args.dataset not in dataset_map:
    raise ValueError(f"未知的数据集：{args.dataset}")


# 获取数据集加载函数或类
dataset_func = dataset_map[args.dataset]
dataset_info = dataset_func(args.data_path)
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = dataset_info

# 定义函数：提取特征
def extract_features(dataloader, model, device):
    """
    从数据加载器中提取特征
    """
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # 提取特征
            features.append(outputs.cpu().numpy())
    return np.concatenate(features, axis=0)

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def compute_wasserstein_distance_custom(X_train, X_test):
    """
    使用 scipy 实现 Wasserstein 距离
    """
    dist_matrix = cdist(X_train, X_test, metric='euclidean')  # 计算距离矩阵
    row_ind, col_ind = linear_sum_assignment(dist_matrix)  # 求解最优分配问题
    wasserstein_dist = dist_matrix[row_ind, col_ind].sum() / len(row_ind)  # 计算平均距离
    return wasserstein_dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 如果数据集是covtype，随机选择train和test的0.1的数据
if args.dataset == 'covtype':
    train_tmp = Subset(dst_train, random.sample(range(len(dst_train)), int(len(dst_train)*0.1)))
    test_tmp = Subset(dst_test, random.sample(range(len(dst_test)), int(len(dst_test)*0.1)))
    train_loader = DataLoader(train_tmp, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_tmp, batch_size=128, shuffle=False)
else:
    # 加载数据
    train_loader = DataLoader(dst_train, batch_size=128, shuffle=False)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False)

# 定义特征提取模型
if args.model == 'ResNet18':
    model = ResNet18(num_classes=num_classes,channel=channel,im_size=im_size).to(device)
elif args.model == 'MLP':
    model = MLP(num_classes=num_classes,channel=channel,im_size=im_size).to(device)
elif args.model == 'TextMLP':
    model = TextMLP(num_classes=num_classes,channel=channel,im_size=im_size).to(device)





# 提取训练集和测试集的特征
train_features = extract_features(train_loader, model, device=device)
test_features = extract_features(test_loader, model, device=device)

# 计算 Wasserstein 距离
wasserstein_distance = compute_wasserstein_distance_custom(train_features, test_features)
print(f"Wasserstein Distance between Train and Test: {wasserstein_distance:.4f}")

from scipy.stats import entropy

def compute_kl_divergence(p, q):
    """
    计算 KL 散度
    :param p: 高维空间的概率分布
    :param q: 低维空间的概率分布
    :return: KL 散度
    """
    p = np.clip(p, 1e-10, None)  # 避免 log(0)
    q = np.clip(q, 1e-10, None)
    kl_div = entropy(p, q)
    return kl_div

def compute_KL_divergence(train_features, test_features):
    """
    计算 KL 散度，确保 train_features 和 test_features 样本数量一致
    """
    # 确保样本数量一致
    n_samples = min(len(train_features), len(test_features))
    train_features = train_features[:n_samples]
    test_features = test_features[:n_samples]

    # 计算高维分布的概率
    p_dist = np.exp(-np.linalg.norm(train_features[:, None] - train_features[None, :], axis=-1))
    p = p_dist / np.sum(p_dist)

    # 计算低维分布的概率
    q_dist = np.exp(-np.linalg.norm(test_features[:, None] - test_features[None, :], axis=-1))
    q = q_dist / np.sum(q_dist)

    # 计算 KL 散度
    kl_divergence = compute_kl_divergence(p.flatten(), q.flatten())
    return kl_divergence

def compute_KL_divergence_cross(train_features, test_features):
    """
    计算 KL 散度，基于 train_features 和 test_features 的交互分布
    """
    # 计算联合分布的概率
    p_dist = np.exp(-np.linalg.norm(train_features[:, None] - test_features[None, :], axis=-1))
    p = p_dist / np.sum(p_dist)

    # 使用相同的方法计算目标分布（这里假设目标分布和联合分布一致）
    q = p  # 如果没有明确目标分布，可以用 p 作为 q

    # 计算 KL 散度
    kl_divergence = compute_kl_divergence(p.flatten(), q.flatten())
    return kl_divergence

def compute_KL_divergence_simple(train_features, test_features):
    """
    简化版 KL 散度计算
    """
    # 转化为概率向量
    p = np.mean(train_features, axis=0)
    q = np.mean(test_features, axis=0)

    # 归一化概率向量
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 计算 KL 散度
    kl_divergence = compute_kl_divergence(p, q)
    return kl_divergence

kl_distance = compute_KL_divergence_simple(train_features, test_features)
print(f"KL Divergence (Simple): {kl_distance:.4f}")

kl_distance = compute_KL_divergence_cross(train_features, test_features)
print(f"KL Divergence (Cross Distribution): {kl_distance:.4f}")

kl_distance = compute_KL_divergence(train_features, test_features)
print(f"KL Divergence: {kl_distance:.4f}")


from scipy.spatial.distance import cdist
import numpy as np

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two datasets X and Y.
    
    Parameters:
        X (ndarray): Data from the first domain (n_samples_X, n_features).
        Y (ndarray): Data from the second domain (n_samples_Y, n_features).
        kernel (str): Kernel type to use ('rbf' or 'linear').
        gamma (float): Kernel bandwidth for the RBF kernel.

    Returns:
        float: MMD value between X and Y.
    """
    # Compute the kernel function
    if kernel == 'rbf':
        K_XX = np.exp(-gamma * cdist(X, X, metric='sqeuclidean'))  # K(X, X)
        K_YY = np.exp(-gamma * cdist(Y, Y, metric='sqeuclidean'))  # K(Y, Y)
        K_XY = np.exp(-gamma * cdist(X, Y, metric='sqeuclidean'))  # K(X, Y)
    elif kernel == 'linear':
        K_XX = X @ X.T
        K_YY = Y @ Y.T
        K_XY = X @ Y.T
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf' or 'linear'.")

    # Compute MMD
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd

# Compute MMD between train_features and test_features
mmd_rbf = compute_mmd(train_features, test_features, kernel='rbf', gamma=0.5)
print(f"MMD (RBF Kernel): {mmd_rbf:.4f}")

mmd_linear = compute_mmd(train_features, test_features, kernel='linear')
print(f"MMD (Linear Kernel): {mmd_linear:.4f}")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def compute_pad(train_features, test_features):
    """
    使用 Proxy A-Distance (PAD) 方法计算源领域和目标领域之间的分布差异
    :param train_features: 源领域的特征
    :param test_features: 目标领域的特征
    :return: Proxy A-Distance (PAD)
    """
    # 准备数据和标签
    X = np.concatenate([train_features, test_features], axis=0)
    y = np.concatenate([np.zeros(len(train_features)), np.ones(len(test_features))])  # 0 为源领域，1 为目标领域

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 使用线性 SVM 训练分类器
    classifier = SVC(kernel="linear", random_state=42)
    classifier.fit(X_train, y_train)

    # 在测试集上评估分类器准确率
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 计算错误率和 PAD
    error_rate = 1 - accuracy
    pad = 2 * (1 - 2 * error_rate)

    return pad

# 使用训练和测试特征计算 PAD
pad_value = compute_pad(train_features, test_features)
print(f"Proxy A-Distance (PAD): {pad_value:.4f}")

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("End time:", end_time)

