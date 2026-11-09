import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
# 无 GPU 时 torch.cuda.set_device(0) 会直接抛 "Found no NVIDIA driver"，
# 连 --help 都跑不起来。加可用性判断：有卡行为不变，无卡退化为不设置。
if torch.cuda.is_available():
    torch.cuda.set_device(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False  # 确保可以使用更多的 cuDNN 算法


import time
import argparse
import numpy as np
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from mydatasets import adult, imdb, CIFAR10,news,covtype,MNIST,CIFAR100,Number,Mnist2MnistM,HR,House,IMDBLarge,TinyImageNet

# ---- 9 个目标数据集统一改走 *_modified 加载器 --------------------------------
# 不用 mydatasets/__init__.py 里的同名函数的三个理由：
#   1) 原始脚本有 bug：cifar100.py 把 num_classes 写死成 10；covtype.py 指向作者
#      机器上的绝对路径；tinyimagenet.py 没把 JPEG 字节解成 RGB。
#      *_modified.py 在外层修掉，且不改原文件一行。
#   2) val/test 口径：*_modified 把官方 test 按类对半切成 val/test。主实验必须与
#      预处理产物、三个 baseline 看到完全相同的划分，否则核心集实验不可比。
#   3) 位置传参陷阱：本脚本用 dataset_func(args.data_path, args.balance_target)
#      位置传参，而原始 MNIST 的签名是 MNIST(data_path, permuted=False, ...)，
#      --balance_target 1 会被绑到 permuted 上，**静默启用像素置换**。
#      *_modified 统一为 (data_path, balance=True)，第 2 个位置参数无害。
# 用同名重新 import，后面的 dataset_map 自动指向 modified 版本。
from mydatasets.cifar100_modified import CIFAR100              # noqa: F811
from mydatasets.covtype_modified import covtype               # noqa: F811
from mydatasets.HR_modified import HR                         # noqa: F811
from mydatasets.tinyimagenet_modified import TinyImageNet     # noqa: F811
from mydatasets.mnist2mnistm_modified import Mnist2MnistM     # noqa: F811
# 以下 4 个此前完全没登记：预处理通了，但主程序加载不了
from mydatasets.mnist_modified import MNIST                   # noqa: F811
from mydatasets.qmnist_modified import QMNIST
from mydatasets.fashionmnist_modified import FashionMNIST
from mydatasets.svhn_modified import SVHN
# -----------------------------------------------------------------------------
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
from collections import Counter

methods = ['kmeans', 'herding','cdal','dvrl','craig','gradmatch','crest',
           'glister','cg_influence','LiSSA','Arnoldi','TracIn',
           'TMCS','KNN_shapley','cs_shapley','mcal','dfal','boundary_aware','svp_EP','mean_std','max_variation_ratio','dbal',
           'lowdimention','max_entropy','kcentergreedy','IAS','random','full']

datasets = ['adult', 'imdb', 'cifar10',"cifar100", 'news', 'covtype','number','mnist2mnistm','HR','House','IMDBLarge','tinyimagenet', 'mnist', 'qmnist', 'fashionmnist', 'SVHN']
# 定义命令行参数
parser = argparse.ArgumentParser(description="Core Set Selection with Herding and KMeans Methods")
parser.add_argument('--data_path', type=str, required=True, help='路径到数据集')
parser.add_argument('--dataset', type=str, choices=datasets, required=True, help='数据集名称')
parser.add_argument('--model', type=str, default='ResNet18', help='模型名称')
parser.add_argument('--method', type=str, choices=methods, required=True, help='选择方法')
parser.add_argument('--selection_optimizer', type=str, default='SGD', help='优化器名称')
parser.add_argument('--selection_lr', type=float, default=0.1, help='学习率')
parser.add_argument('--selection_momentum', type=float, default=0.9, help='动量因子')
parser.add_argument('--selection_weight_decay', type=float, default=5e-4, help='权重衰减')
parser.add_argument('--selection_nesterov', type=bool, default=True, help='是否使用 Nesterov 动量')
parser.add_argument('--selection_batch', type=int, default=128, help='批大小')
parser.add_argument('--selection_fraction', type=float, default=0.1, help='核心集比例')
parser.add_argument('--print_freq', type=int, default=10, help='打印频率')
parser.add_argument('--gpu', type=int, default=0, help='使用的 GPU 设备')
parser.add_argument('--epochs', type=int, default=100, help='训练的轮数')
parser.add_argument('--workers', type=int, default=4, help='数据加载器的工作线程数')
parser.add_argument('--stepsize', type=int, default=10, help='学习率调度器的步长')
parser.add_argument('--gamma', type=float, default=0.1, help='学习率调度器的衰减因子')
parser.add_argument('--proxy_model', type=str, default='ProxyMLP', help='代理模型名称')
parser.add_argument('--proxy_epochs', type=int, default=20, help='代理模型训练的轮数')
parser.add_argument('--proxy_lr', type=float, default=0.01, help='代理模型的学习率')
parser.add_argument('--active_fraction_ratio', type=int, default=0.1, help='代理模型的批大小')
parser.add_argument('--random_seed', type=int, default=0, help='随机种子')
parser.add_argument('--Is_pretrain', type=bool, default=True, help='是否预训练')
parser.add_argument('--pretrain_epochs', type=int, default=3, help='预训练的轮数')
parser.add_argument('--dve_lr', type=float, default=0.001, help='数据价值评估器的学习率')
parser.add_argument('--selection_interval', type=int, default=5, help='选择间隔')
parser.add_argument('--lambda_reg', type=float, default=0.5, help='选择间隔')
parser.add_argument('--balance', type=bool, default=True, help='是否平衡选择')
parser.add_argument('--save_best_model_path', type=str, default='./best_models/', help='最佳模型保存路径')
parser.add_argument('--rl_select_size', type=int, default=500, help='RL选择大小')
parser.add_argument('--balance_target', type=int, default=1, help='是否目标分布平衡')
parser.add_argument('--selection_framework', type=str, default='once', help='选择选择数据的框架')
parser.add_argument('--save_log_path', type=str, default='./logs/', help='日志保存路径')
parser.add_argument('--patience', type=int, default=10, help='早停参数')
parser.add_argument('--task_type', type=str,default='classification', help='任务类型')

# 解析初始参数
args, remaining_args = parser.parse_known_args()

# 根据方法动态添加选项
if args.method == 'mcal':
    parser.add_argument('--uncertainty_type', type=str, choices=["BvSB","max_entropy","max_variation_ratio","mean_std"], default='BvSB', help='不确定性类型')
elif args.method == 'TMCS':
    parser.add_argument('--shapley_util', type=str, choices=["model","feature","uncertainty"], default='feature', help='Shapley值计算方法')

# 重新解析包含新增选项的所有参数
args = parser.parse_args()

# 数据集映射
dataset_map = {
    'adult': adult,
    'imdb': imdb,
    'cifar10': CIFAR10,
    'news': news,
    'covtype': covtype,
    'cifar100': CIFAR100,
    'number': Number,
    'mnist2mnistm': Mnist2MnistM,
    'HR': HR,
    'House': House,
    'IMDBLarge': IMDBLarge,
    'tinyimagenet': TinyImageNet,
    'mnist': MNIST,
    'qmnist': QMNIST,
    'fashionmnist': FashionMNIST,
    'SVHN': SVHN,
}

start_time = time.time()
# 检查 dataset 是否在映射中
if args.dataset not in dataset_map:
    raise ValueError(f"未知的数据集：{args.dataset}")

# 获取数据集加载函数或类
dataset_func = dataset_map[args.dataset]
dataset_info = dataset_func(args.data_path, args.balance_target)
print(f"Target distribution balanced: {args.balance_target}")


channel, im_size, num_classes, class_names, mean, std, dst_train,dst_val, dst_test = dataset_info
run_method(args,channel,im_size,num_classes,class_names,mean,std, dst_train, dst_val, dst_test)



end_time = time.time()
print(f"总耗时：{end_time - start_time:.2f} 秒")