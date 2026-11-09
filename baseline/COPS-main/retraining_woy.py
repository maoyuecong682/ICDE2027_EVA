import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import npy_dataset
import argparse
import numpy as np
from models.resnet import *
import os
import random
from models.simple_cnn import Vanillann
from models.densenet import DenseNet121
from models.mobilenet import MobileNetV2
import adult_adapter  # Adult 表格数据适配层
import eva_adapter as EVA  # EVA 预处理数据的统一适配层

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', type=str, default='resnet20', help="name of save log") 
parser.add_argument('--optimizer', type=str, default='adamw', help="adamw/sgd") 
parser.add_argument('--dataset', type=str, default='cifar', help="cifar/svhn") 
parser.add_argument('--test_mode', type=str, default='random') 
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--select-only', action='store_true', help='Save selection without native retraining')
parser.add_argument('--T_max', type=int, default=50) 
parser.add_argument('--eta_min', default=0, type=float, help='min lr')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--weight_decay', type=float, default=5e-4) 
parser.add_argument('--sample_number', type=int, default=300) 
parser.add_argument('--schedule_type', type=str, default="cosine", help="step/cosine/poly") 
parser.add_argument('--loss_type', type=str, default="base", help="base/reweight_clip") 
parser.add_argument('--weight_folder', type=str, default='active_weight/') 
parser.add_argument('--weight_name', type=str, default='active_weight.pth') 
parser.add_argument('--constant_1', type=float, default=0.005) 
parser.add_argument('--constant_2', type=float, default=0) 
parser.add_argument('--un_folder', type=str, default='cifar_npy') 
parser.add_argument('--batch_size', type=int, default=0, help='0 表示保持原脚本常量；CPU/小内存时用来压激活值')
args = parser.parse_args()
BS_TRAIN = args.batch_size if args.batch_size > 0 else 256
BS_EVAL = args.batch_size if args.batch_size > 0 else 128

def cal_un_woy(logit):
    covariance = np.matmul((logit-np.mean(logit, axis=0, keepdims=True)).transpose()[1:, :], (logit-np.mean(logit, axis=0, keepdims=True))[:, 1:])/max(len(logit)-1, 1)
    
    logit =logit -np.max(logit, 1, keepdims=True)
    pred = np.mean(np.exp(logit)/np.sum(np.exp(logit), 1,  keepdims=True), 0)[1:]

    pred_ = np.reshape(pred, (len(pred), 1))
    psi = np.diag(pred) - np.matmul(pred_, pred_.transpose())
   # print(logit.shape)
   # print(pred.shape)
   # print(psi.shape)
   # print(covariance.shape)
   # ssss
    met = np.trace(np.matmul(psi, covariance))
    return met

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
setup_seed(args.seed)
os.makedirs(args.weight_folder, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IS_TABULAR = False
IS_EVA = False

if args.dataset == "cifar100" and args.dataset not in EVA.ADAPTER_DATASETS:
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
    train_data =np.load("cifar100_npy/train_X_2.npy")
    train_lab = np.load("cifar100_npy/train_Y_2.npy")
    test_data =np.load( "cifar100_npy/train_X_1.npy")
    test_lab = np.load("cifar100_npy/train_Y_1.npy")
    real_test_data =np.load( "cifar100_npy/test_X.npy")
    real_test_lab = np.load("cifar100_npy/test_Y.npy")
    train_lab_oh = np.zeros((len(train_lab), 100))
    for k in range(len(train_lab_oh)):
        train_lab_oh[k, train_lab[k]] = 1
    train_logit = np.load(os.path.join("cifar100_npy", "train_UN_2_raw.npy"))
    # train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    # print(train_pseudo_label)
    # for j in range(100):
    #     print(len([i for i in range(len(train_pseudo_label)) if train_pseudo_label[i]==j]))

    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
   # print(len(train_logit))
    for k in range(len(train_logit)):
    #    print(k)
        train_metric[k] = cal_un_woy(train_logit[k] )


elif args.dataset == "cifar":
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
    train_data =np.load("cifar_npy/train_X_2.npy")
    train_lab = np.load("cifar_npy/train_Y_2.npy")
    test_data =np.load( "cifar_npy/train_X_1.npy")
    test_lab = np.load("cifar_npy/train_Y_1.npy")
    real_test_data =np.load( "cifar_npy/test_X.npy")
    real_test_lab = np.load("cifar_npy/test_Y.npy")
    train_logit = np.load(os.path.join('cifar_npy', "train_UN_2_raw.npy"))
    # train_pseudo_label = np.argmax(np.mean(train_logit, 0), 1)
    # print(train_pseudo_label)
    # for j in range(10):
    #     print(len([i for i in range(len(train_pseudo_label)) if train_pseudo_label[i]==j]))

    train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k] )



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
    train_data =np.load("cifar_noise_npy/train_X_2.npy")
    train_lab = np.load("cifar_noise_npy/train_Y_2_aggre.npy")
    test_data =np.load( "cifar_noise_npy/train_X_1.npy")
    test_lab = np.load("cifar_noise_npy/train_Y_1_aggre.npy")
    real_test_data = np.load( "cifar_noise_npy/test_X.npy")
    real_test_lab = np.load("cifar_noise_npy/test_Y.npy")
    train_logit = np.load(os.path.join("cifar_aggre_npy", "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1,0,2))
    # train_logit = train_logit.transpose((1,0,2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k] )
    
elif args.dataset == 'adult' and args.dataset not in EVA.ADAPTER_DATASETS:
    # ---------------- Adult 表格数据（COPS without y） ----------------
    class_n = 2
    IS_TABULAR = True
    transform_train = None
    transform_test = None
    un_dir = args.un_folder
    train_data = np.load(os.path.join(un_dir, "train_X_2.npy")).astype(np.float32)
    train_lab = np.load(os.path.join(un_dir, "train_Y_2.npy")).astype(np.int64)
    test_data = np.load(os.path.join(un_dir, "train_X_1.npy")).astype(np.float32)
    test_lab = np.load(os.path.join(un_dir, "train_Y_1.npy")).astype(np.int64)
    real_test_data = np.load(os.path.join(un_dir, "test_X.npy")).astype(np.float32)
    real_test_lab = np.load(os.path.join(un_dir, "test_Y.npy")).astype(np.int64)

    train_logit = np.load(os.path.join(un_dir, "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1, 0, 2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k])
    print('[adult-woy] sampling=%d probe=%d test=%d  metric min=%.6e max=%.6e'
          % (len(train_data), len(test_data), len(real_test_data),
             train_metric.min(), train_metric.max()))

elif args.dataset in EVA.DATASETS:
    # ---------------- EVA 预处理好的 9 个数据集（COPS without y） ----------------
    IS_EVA = True
    _probe, _sampling, _test, _meta = EVA.cops_views(args.dataset)
    IS_TABULAR = bool(_meta['is_tabular'])
    class_n = _meta['num_classes']
    transform_train = None
    transform_test = None
    train_data = _sampling
    train_lab = _sampling.labels
    test_data = _probe
    test_lab = _probe.labels
    real_test_data = _test
    real_test_lab = _test.labels

    train_logit = np.load(os.path.join(args.un_folder, "train_UN_2_raw.npy"))
    train_logit = train_logit.transpose((1, 0, 2))
    train_metric = np.zeros((len(train_logit), ))
    for k in range(len(train_logit)):
        train_metric[k] = cal_un_woy(train_logit[k])
    print('[%s-woy] sampling=%d probe=%d test=%d classes=%d  metric min=%.6e max=%.6e'
          % (args.dataset, len(train_data), len(test_data), len(real_test_data), class_n,
             train_metric.min(), train_metric.max()))

######################### sample_selection
_pos = train_metric[train_metric > 0]
if len(_pos) > 0:
    train_metric[train_metric <= 0] = np.min(_pos)
else:
    train_metric = np.abs(train_metric) + 1e-8
train_metric = np.sqrt(train_metric)


def _sample_probs(metric_clipped, ind):
    """归一化采样概率；全 0 时退回均匀分布，避免 NaN。"""
    w = metric_clipped[ind]
    s = np.sum(w)
    if (not np.isfinite(s)) or s <= 0:
        return None
    p = np.clip(w, 0, None)
    return p / p.sum()



######################### sample_selection

if args.test_mode == 'random':
    train_ind = []
    train_loss_weight = []
    ind =  [i for i in range(len(train_metric))]
    tmp_ind = np.random.choice(ind, size=(args.sample_number*class_n, ),replace=False)
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)

elif args.test_mode == 'oracle_sampling_cut':
    train_metric_ = np.clip(train_metric,a_min=0, a_max=args.constant_1)
    train_ind = []
    train_loss_weight = []
    ind =  [i for i in range(len(train_metric)) ]
    tmp_ind = np.random.choice(ind, size=(args.sample_number*class_n, ),replace=False, p=_sample_probs(train_metric_, ind))
    train_loss_weight.append(train_metric[tmp_ind])
    train_ind.extend(tmp_ind)





#####################################
if IS_EVA:
    train_data = EVA.select(train_data, train_ind)   # 按索引取核心集，不复制像素
    train_lab = train_data.labels
else:
    train_data = train_data[train_ind]
    train_lab = train_lab[train_ind]

train_loss_weight = np.concatenate(train_loss_weight)

# 核心集索引必须落盘：下游统一评估器要用它复现同一个子集
_sel = np.asarray(train_ind, dtype=np.int64)
np.save(os.path.join(args.weight_folder, "selected_indices.npy"), _sel)
if IS_EVA:
    np.save(os.path.join(args.weight_folder, "selected_train_indices.npy"), np.asarray(train_data.indices))
if args.select_only:
    raise SystemExit(0)
print('[coreset] saved %d indices -> %s'
      % (len(_sel), os.path.join(args.weight_folder, "selected_indices.npy")))


################################


###########################################

def _make_dataset(x, y, tf):
    """表格数据用 AdultDataset，EVA 数据用适配层已包装好的 Dataset，图像仍走 NPY_Dataset。"""
    if IS_EVA:
        return x
    if IS_TABULAR:
        return adult_adapter.AdultDataset(x, y)
    return npy_dataset.NPY_Dataset(x, y, tf)


# EVA 的 Wrapped/Indexed 自带 __len__，但 2GB 配额下多进程 worker 会复制一份数据集，
# 所以和表格数据一样强制单进程。
# 图像走多进程解包；表格保持单进程（实测 covtype 上 4 worker 比 0 worker 慢 2.6 倍）
_NW = 0 if IS_TABULAR or not torch.cuda.is_available() else 4
if  args.loss_type !=  "base":
    if IS_EVA:
        # 核心集样本 + 逐样本权重；三元素协议与 NPY_Dataset_2 / _WeightedAdult 一致
        trainset = EVA.Weighted(train_data, train_loss_weight)
    elif IS_TABULAR:
        class _WeightedAdult(adult_adapter.AdultDataset):
            def __init__(self, x, y, w):
                super(_WeightedAdult, self).__init__(x, y)
                self.w = np.asarray(w, dtype=np.float32)

            def __getitem__(self, i):
                x, y = super(_WeightedAdult, self).__getitem__(i)
                return x, y, float(self.w[i])

        trainset = _WeightedAdult(train_data, train_lab, train_loss_weight)
    else:
        trainset = npy_dataset.NPY_Dataset_2(train_data, train_lab,train_loss_weight,  transform_train)
else:
    trainset = _make_dataset(train_data, train_lab, transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BS_TRAIN, shuffle=True, num_workers=_NW)
testset = _make_dataset(train_data, train_lab, transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BS_EVAL, shuffle=False, num_workers=_NW)
testset_2 = _make_dataset(test_data, test_lab, transform_test)
testloader_2 = torch.utils.data.DataLoader(
    testset_2, batch_size=BS_EVAL, shuffle=False, num_workers=_NW)
testset_3 = _make_dataset(real_test_data, real_test_lab, transform_test)
testloader_3 = torch.utils.data.DataLoader(
    testset_3, batch_size=BS_EVAL, shuffle=False, num_workers=_NW)

print('==> Building model..')


def train(epoch,  net,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if  args.loss_type !=  "base": 
        criterion = nn.CrossEntropyLoss(reduction='none')
        for batch_idx, (inputs, targets,tr_weight) in enumerate(trainloader):
            inputs, targets, tr_weight = inputs.to(device), targets.to(device), tr_weight.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            t_loss = criterion(outputs, targets.long())
       #     pdb.set_trace()
            if args.loss_type ==  "reweight_clip": 
                rw = torch.clip(1/tr_weight , max=args.constant_2)
                loss = torch.sum(t_loss * rw / torch.sum(rw))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # _, predicted = outputs.max(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Training Loss : ', train_loss /len(trainloader))
    else:
        criterion = nn.CrossEntropyLoss()
        for batch_idx, (inputs, targets,) in enumerate(trainloader):
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
    # print(print_name, test_loss/len(testloader))
    print(print_name + ' Acc : ', acc)
    return acc


## 记录输出的logit

# torch.torch.manual_seed(0)

if args.model == 'resnet20':
    net = ResNet20(num_classes=class_n)
elif args.model == 'resnet32':
    net = ResNet32(num_classes=class_n)
elif args.model == 'resnet44':
    net = ResNet44(num_classes=class_n)
elif args.model == 'resnet56':
    net = ResNet56(num_classes=class_n)
elif args.model == 'simplecnn':
    net = Vanillann()
elif args.model == 'mobilev2':
    net = MobileNetV2(num_classes=class_n)
elif args.model == 'dense121':
    net = DenseNet121()
elif args.dataset in EVA.ADAPTER_DATASETS:
    # 同 retraining_withy：EVA 数据集优先，arch=None 由适配层按形态自选骨干。
    net = EVA.build_model(args.dataset, arch=None, num_classes=class_n)
elif args.model in ('tabnet', 'mlp'):
    net = adult_adapter.build_model(
        args.model, input_features=train_data.shape[1], num_classes=class_n)


net = net.to(device)


criterion = nn.CrossEntropyLoss()
if args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
if args.schedule_type == "step":
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=args.gamma)
elif args.schedule_type == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
for epoch in range(args.epochs):
    train(epoch,  net,trainloader)
    scheduler.step()
    test(epoch, net,testloader, 'Train Loss')
    test(epoch, net,testloader_3, 'Real Test Loss')
