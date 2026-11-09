import os
import traceback
import torch

# 导入你的所有数据集模块（根据你的 __init__.py 和文件列表）
from adult import adult
from bike import Bike
from cifar10 import CIFAR10
from cifar100 import CIFAR100
from covtype import covtype
from fashionmnist import FashionMNIST
from imagenet import ImageNet
from imdb_large import IMDBLarge
from imdb0 import imdb
from mnist import MNIST
from mnist2mnistm import Mnist2MnistM
from news import news
from number import Number
from qmnist import QMNIST
from svhn import SVHN
from tinyimagenet import TinyImageNet

# 你的统一数据根目录
DATA_ROOT = "/root/autodl-tmp/EVA/data"

def test_dataset(name, func, *args, **kwargs):
    print(f"\n{'='*40}")
    print(f"正在测试: {name}")
    try:
        # 调用数据集加载函数
        result = func(*args, **kwargs)
        
        # 大多数返回 9 个元素，有些返回 8 个
        train_ds = result[-3]
        val_ds = result[-2] if len(result) >= 9 else "N/A"
        test_ds = result[-1]
        
        print(f"✅ {name} 加载成功！")
        print(f"   Train 样本数: {len(train_ds)}")
        if val_ds != "N/A":
            print(f"   Val   样本数: {len(val_ds)}")
        print(f"   Test  样本数: {len(test_ds)}")
    except Exception as e:
        print(f"❌ {name} 加载失败！")
        print(f"   错误信息: {e}")
        # 打印具体的报错行，方便你排查路径问题
        # traceback.print_exc() 

if __name__ == "__main__":
    # 这里只测试代码逻辑和路径，如果本地没有大数据集，把不存在的注释掉
    # AutoDL 本地已有的自动下载数据集
    test_dataset("FashionMNIST", FashionMNIST, DATA_ROOT)
    test_dataset("MNIST", MNIST, DATA_ROOT)
    test_dataset("QMNIST", QMNIST, DATA_ROOT)
    test_dataset("SVHN", SVHN, DATA_ROOT)
    
    # 需要你提前上传的数据集
    test_dataset("Adult", adult, DATA_ROOT)
    test_dataset("Bike", Bike, DATA_ROOT)
    test_dataset("CIFAR10", CIFAR10, DATA_ROOT + "/cifar10")
    test_dataset("CIFAR100", CIFAR100, DATA_ROOT + "/cifar100")
    test_dataset("Covtype", covtype, DATA_ROOT)
    test_dataset("IMDBLarge", IMDBLarge, DATA_ROOT) # 内部路径已写死，传参不影响
    test_dataset("TinyImageNet", TinyImageNet, DATA_ROOT) # 内部路径已写死
    
    # 需要额外测试的文本和领域自适应数据集 (极度耗时/吃内存，建议有显卡再跑)
    # test_dataset("IMDB (BERT)", imdb, DATA_ROOT)
    # test_dataset("News (BERT)", news, DATA_ROOT)
    # test_dataset("Mnist2MnistM", Mnist2MnistM, DATA_ROOT)
    # test_dataset("Number", Number, DATA_ROOT)
    
    print("\n" + "="*40)
    print("测试完毕！")
