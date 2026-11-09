import torch
import torch.nn as nn
import torch.optim as optim
from utils.train_utils import get_pred_raw
import torchvision.transforms as transforms
import npy_dataset
import argparse
import numpy as np
import copy
from models.resnet import *
import os
import random
from models.simple_cnn import Vanillann
from models.densenet import DenseNet121
from models.mobilenet import MobileNetV2
import adult_adapter  # Adult 表格数据适配层
import eva_adapter as EVA  # EVA 预处理数据的统一适配层（与 GM / DUAL 共用同一份）

def descent_lr_train(lr, epoch, optimizer, interval):
    for k in interval:
        if epoch < k:
            break
        else:
            lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', type=str, default='cifar10') 
parser.add_argument('--model', type=str, default='resnet20') 
parser.add_argument('--model_num', type=int, default=10) 
parser.add_argument('--num_classes', type=int, default=10) 
parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--selection-only', action='store_true', help='Only train ensemble and export sampling logits; skip diagnostic test inference')
parser.add_argument('--weight_folder', type=str, default='generation_weight_cifar10_resnet20/') 
parser.add_argument('--npy_folder', type=str, default='cifar10_npy_resnet20/') 
parser.add_argument('--batch_size', type=int, default=0, help='0 表示保持原脚本常量；CPU/小内存时用来压激活值')
args = parser.parse_args()
BS_EVAL = args.batch_size if args.batch_size > 0 else 128
os.makedirs(args.weight_folder, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IS_TABULAR = False  # Adult 等表格数据走不走图像 transform / PIL 分支
IS_EVA = False      # 走 eva_adapter 的数据集：不走 npy、不 materialize
if args.dataset == 'cifar10' and args.dataset not in EVA.ADAPTER_DATASETS:
    class_n = 10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_npy/train_X_1.npy")
    train_lab = np.load("cifar_npy/train_Y_1.npy")
    test_data =np.load( "cifar_npy/train_X_2.npy")
    test_lab = np.load("cifar_npy/train_Y_2.npy")
    real_test_data =np.load( "cifar_npy/test_X.npy")
    real_test_lab = np.load("cifar_npy/test_Y.npy")
    print(len(train_data), len(test_data), len(real_test_data))
elif args.dataset == "cifar100" and args.dataset not in EVA.ADAPTER_DATASETS:
    # cifar100 交给适配层：原生测试集 10000 张，EVA 的划分是 val5000/test5000
    class_n = 100
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar100_npy/train_X_1.npy")
    train_lab = np.load("cifar100_npy/train_Y_1.npy")
    test_data =np.load( "cifar100_npy/train_X_2.npy")
    test_lab = np.load("cifar100_npy/train_Y_2.npy")
    real_test_data =np.load( "cifar100_npy/test_X.npy")
    real_test_lab = np.load("cifar100_npy/test_Y.npy")
    print(len(train_data), len(test_data), len(real_test_data))

elif args.dataset == "cifar10_aggre":
    class_n = 10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    train_data =np.load("cifar_noise_npy/train_X_1.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_1_aggre.npy")
    test_data =np.load( "cifar_noise_npy/train_X_2.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_2_aggre.npy")
    real_test_data =np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    print(len(train_data), len(test_data), len(real_test_data))

elif args.dataset == 'adult' and args.dataset not in EVA.ADAPTER_DATASETS:
    # ---------------- Adult 表格数据 ----------------
    # 只对训练类别探测，不改变 COPS 算法；数据由 prepare_adult_npy.py 生成
    class_n = args.num_classes
    IS_TABULAR = True
    transform_train = None
    transform_test = None
    train_data = np.load(os.path.join(args.npy_folder, "train_X_1.npy")).astype(np.float32)
    train_lab = np.load(os.path.join(args.npy_folder, "train_Y_1.npy")).astype(np.int64)
    test_data = np.load(os.path.join(args.npy_folder, "train_X_2.npy")).astype(np.float32)
    test_lab = np.load(os.path.join(args.npy_folder, "train_Y_2.npy")).astype(np.int64)
    real_test_data = np.load(os.path.join(args.npy_folder, "test_X.npy")).astype(np.float32)
    real_test_lab = np.load(os.path.join(args.npy_folder, "test_Y.npy")).astype(np.int64)
    print('[adult] probe=%d sampling=%d test=%d feat=%d'
          % (len(train_data), len(test_data), len(real_test_data), train_data.shape[1]))

elif args.dataset in EVA.DATASETS:
    # ---------------- EVA 预处理好的 9 个数据集 ----------------
    # 刻意**不落 npy**：直接用 eva_adapter 的惰性包装。
    #   1) 预处理与 EVA 主实验逐位一致（同一份 *_modified.py、同一份 mean/std）；
    #   2) cgroup 上限只有 2GB，materialize 整份图像（uint8 也要 150-300MB/份，
    #      且 COPS 要 5 份）会把进程顶死。
    # probe / sampling 的 20-80 按类切分由 eva_adapter.cops_views 统一负责，
    # 与 adult 那次（probe_ratio=0.2, seed=48）完全同口径。
    IS_EVA = True
    _probe, _sampling, _test, _meta = EVA.cops_views(args.dataset)
    IS_TABULAR = bool(_meta['is_tabular'])
    class_n = _meta['num_classes']
    args.num_classes = class_n
    transform_train = None
    transform_test = None
    train_data = _probe            # probe：训练不确定性模型
    train_lab = _probe.labels      # numpy int64，供 np.zeros((M, len, C))
    test_data = _sampling          # sampling 池：COPS 从这里选核心集
    test_lab = _sampling.labels
    real_test_data = _test         # 真实 test
    real_test_lab = _test.labels
    print('[%s] probe=%d sampling=%d test=%d classes=%d tabular=%s'
          % (args.dataset, len(_probe), len(_sampling), len(_test),
             class_n, IS_TABULAR))


def _make_dataset(x, y, tf):
    """表格数据用 AdultDataset，EVA 数据用适配层已包装好的 Dataset，图像仍走 NPY_Dataset。"""
    if IS_EVA:
        return x
    if IS_TABULAR:
        return adult_adapter.AdultDataset(x, y)
    return npy_dataset.NPY_Dataset(x, y, tf)


# 图像走多进程解包；表格保持单进程（实测 covtype 上 4 worker 比 0 worker 慢 2.6 倍）
_NW = 0 if IS_TABULAR or not torch.cuda.is_available() else 4
trainset = _make_dataset(train_data, train_lab, transform_train)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BS_EVAL, shuffle=True, num_workers=_NW)

testset = _make_dataset(train_data, train_lab, transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BS_EVAL, shuffle=False, num_workers=_NW)

testset_2 = _make_dataset(test_data, test_lab, transform_test)
testloader_2 = torch.utils.data.DataLoader(
    testset_2, batch_size=BS_EVAL, shuffle=False, num_workers=_NW)

testset_3 = _make_dataset(real_test_data, real_test_lab, transform_test)
testloader_3 = torch.utils.data.DataLoader(
    testset_3, batch_size=BS_EVAL, shuffle=False, num_workers=_NW)


# Model
print('==> Building model..')

os.makedirs(args.npy_folder, exist_ok=True)
def train(epoch,  net,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Training Loss : ', train_loss /len(trainloader))


def test(epoch, net,testloader, print_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print(print_name, test_loss/len(testloader))
    print(print_name + ' Acc : ', acc)
    return acc


out_logit = None if args.selection_only else np.zeros((args.model_num,len(train_lab), args.num_classes))
out_logit_2 = np.zeros((args.model_num,len(test_lab), args.num_classes))
out_logit_3 = None if args.selection_only else np.zeros((args.model_num,len(real_test_lab), args.num_classes))

for k in range(args.model_num):
    setup_seed(k)
    torch.torch.manual_seed(k)
    if args.model == 'resnet20':
        net = ResNet20(num_classes=args.num_classes)
    elif args.model == 'resnet32':
        net = ResNet32(num_classes=args.num_classes)
    elif args.model == 'resnet44':
        net = ResNet44(num_classes=args.num_classes)
    elif args.model == 'resnet56':
        net = ResNet56(num_classes=args.num_classes)
    elif args.model == 'simplecnn':
        net = Vanillann()
    elif args.model == 'mobilev2':
        net = MobileNetV2(num_classes=args.num_classes)
    elif args.model == 'dense121':
        net = DenseNet121()
    elif args.model in ('tabnet', 'mlp'):
        net = adult_adapter.build_model(
            args.model, input_features=train_data.shape[1], num_classes=args.num_classes)
    elif args.model == 'resnet18':
        # EVA 9 个数据集的统一骨干；表格数据会自动落到 TabMLP
        net = EVA.build_model(_meta, arch=None, num_classes=args.num_classes)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    if IS_TABULAR:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4 )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60, 90], gamma=0.1)
    for epoch in range(args.epochs):
        descent_lr_train(args.lr, epoch+1, optimizer, [30, 60 ,90])
        train(epoch, net, trainloader)
        if not args.selection_only:
            test(epoch, net,testloader, 'Train Loss')
            test(epoch, net,testloader_2, 'Hold Test Loss')
            test(epoch, net,testloader_3, 'Real Test Loss')
    torch.save(net.state_dict(), os.path.join(args.weight_folder, 'uncertainty_generation_'+str(k)+'.pth'))
    if not args.selection_only:
        tmp_logit = get_pred_raw(testloader, net, )
        out_logit[k] = copy.deepcopy(tmp_logit)
    tmp_logit_2 = get_pred_raw(testloader_2, net, )
    out_logit_2[k] = copy.deepcopy(tmp_logit_2)
    if not args.selection_only:
        tmp_logit_3 = get_pred_raw(testloader_3, net, )
        out_logit_3[k] = copy.deepcopy(tmp_logit_3)

if not args.selection_only:
    print(out_logit.shape)
    np.save(os.path.join(args.npy_folder, "train_UN_1_raw.npy"), out_logit)
print(out_logit_2.shape)
np.save(os.path.join(args.npy_folder, "train_UN_2_raw.npy"), out_logit_2)
if not args.selection_only:
    print(out_logit_3.shape)
    np.save(os.path.join(args.npy_folder, "test_UN_raw.npy"), out_logit_3)

# ---- 标签落盘（EVA 分支）----------------------------------------------------
# calibrate_adult_metric.py 需要 train_Y_2 才能算 constant_1；原版只存 logits，
# 缺标签会让标定静默退回 1e9（=不裁剪），COPS 的加权机制就此失效而不报错。
# 命名与 logits 严格对应：Y_1 ↔ UN_1（probe）、Y_2 ↔ UN_2（采样池）、
# test_Y ↔ test_UN（真实测试集）。
# 只在 EVA 分支写：原生分支自己就有 Y 文件（noise 分支还叫 *_aggre.npy），
# 无条件写会被当成「同一目录两套标签」。
if IS_EVA:
    np.save(os.path.join(args.npy_folder, "train_Y_1.npy"), np.asarray(train_lab, dtype=np.int64))
    np.save(os.path.join(args.npy_folder, "train_Y_2.npy"), np.asarray(test_lab, dtype=np.int64))
    np.save(os.path.join(args.npy_folder, "test_Y.npy"), np.asarray(real_test_lab, dtype=np.int64))
    print("[labels] saved train_Y_1 / train_Y_2 / test_Y ->", args.npy_folder)

