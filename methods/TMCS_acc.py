import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import copy
from collections import Counter
import nets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from collections import defaultdict

# 定义一个字典来存储钩子捕获的特征
activation = defaultdict(list)


def compute_similarity_matrix(features, device):
    """
    计算特征矩阵的相似度矩阵。

    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        device (str): 设备类型。

    Returns:
        torch.Tensor: 相似度矩阵，形状为 (N, N)。
    """
    # 简单使用内积作为相似度度量
    features = features.to(device)
    similarity_matrix = torch.mm(features, features.t())
    return similarity_matrix

def compute_shapley_value(features, labels, model, device, num_iterations=100):
    """
    使用 Accuracy 作为效用函数计算 Shapley 值（精简版）。
    
    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        labels (torch.Tensor): 标签张量，形状为 (N,)。
        model (torch.nn.Module): 用于评估性能的模型。
        device (str): 设备类型。
        num_iterations (int): Monte Carlo 随机排列次数。
    
    Returns:
        torch.Tensor: 每个样本的 Shapley 值，形状为 (N,)。
    """
    num_samples = features.size(0)
    shapley_values = torch.zeros(num_samples, device=device)
    labels = labels.to(device)

    # 预训练模型一次，避免重复训练
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # 全部样本的初始性能
        outputs = model(features.to(device))
        _, predictions = torch.max(outputs, 1)
        full_accuracy = (predictions == labels).float().mean().item()

    for t in tqdm(range(num_iterations), desc="Calculating Shapley Values"):
        # 随机排列
        permutation = torch.randperm(num_samples, device=device)
        previous_performance = 0.0

        for j, idx in enumerate(permutation):
            subset_indices = permutation[:j + 1]
            subset_features = features[subset_indices]
            subset_labels = labels[subset_indices]

            # 评估当前子集性能
            with torch.no_grad():
                outputs = model(subset_features.to(device))
                _, predictions = torch.max(outputs, 1)
                current_accuracy = (predictions == subset_labels).float().mean().item()

            # 计算边际贡献
            marginal_contribution = current_accuracy - previous_performance
            previous_performance = current_accuracy

            # 更新 Shapley 值
            shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (t + 1)

    # 归一化 Shapley 值
    shapley_values = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min())
    return shapley_values

# def tmcs_shapley_selection(args,features, labels, model, target_core_set_size, device):
#     """
#     使用 Shapley Value 选择核心数据集。

#     Args:
#         features (torch.Tensor): 特征张量，形状为 (N, D)。
#         labels (torch.Tensor): 标签张量，形状为 (N,)。
#         target_core_set_size (int): 核心数据集的目标大小。
#         device (str): 设备类型。

#     Returns:
#         List[int]: 核心数据集的索引。
#     """
#     shapley_values = compute_shapley_value(features, labels,model, device)
#     if args.target_balance:
#         pass
#     else:
#      selected_indices = torch.topk(shapley_values, target_core_set_size).indices.tolist()
#     return selected_indices


def tmcs_shapley_selection(args, features, labels, model, target_core_set_size, device):
    """
    使用 Shapley Value 选择核心数据集。

    Args:
        args (Namespace): 包含各种参数的命名空间。
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        labels (torch.Tensor): 标签张量，形状为 (N,)。
        model (torch.nn.Module): 用于评估性能的模型。
        target_core_set_size (int): 核心数据集的目标大小。
        device (str): 设备类型。

    Returns:
        List[int]: 核心数据集的索引。
    """
    # 计算 Shapley 值
    shapley_values = compute_shapley_value(features, labels, model, device)
    selected_indices = []

    if args.balance_target:
        # 目标是平衡每个类别的样本数
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        samples_per_class = target_core_set_size // num_classes

        for label in unique_labels:
            # 获取当前类别的索引
            class_indices = (labels == label).nonzero(as_tuple=True)[0]
            class_shapley_values = shapley_values[class_indices]

            # 对当前类别按 Shapley 值排序，选择前 samples_per_class 个样本
            selected_class_indices = torch.topk(class_shapley_values, min(samples_per_class, len(class_indices))).indices
            selected_indices.extend(class_indices[selected_class_indices].tolist())
        
        # 如果还有剩余的样本需要填充
        if len(selected_indices) < target_core_set_size:
            remaining_indices = torch.topk(shapley_values, target_core_set_size - len(selected_indices)).indices
            selected_indices.extend(remaining_indices.tolist())
    else:
        # 如果不需要平衡，直接选择 Shapley 值最高的样本
        selected_indices = torch.topk(shapley_values, target_core_set_size).indices.tolist()

    return selected_indices


# def extract_features_and_labels(model, data_loader, device):
#     model.eval()
#     all_features, all_labels = [], []
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             features = model(inputs)  # 假设模型最后一层输出特征
#             all_features.append(features.cpu())
#             all_labels.append(labels.cpu())
#     return torch.cat(all_features), torch.cat(all_labels)

def extract_features_and_labels(data_loader, device):
    """
    从数据加载器中提取原始特征和标签。

    Args:
        data_loader (DataLoader): 数据加载器。
        device (torch.device): 设备。

    Returns:
        torch.Tensor, torch.Tensor: 原始特征张量和标签张量。
    """
    all_features, all_labels = [], []
    for inputs, labels in data_loader:
        # 不再通过模型提取特征，直接使用原始特征
        all_features.append(inputs.to(device))
        all_labels.append(labels.to(device))
    return torch.cat(all_features), torch.cat(all_labels)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, precision, recall, f1

def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, device, val_loader, best_model_path, patience=10, best_acc=0.0):
    trigger_times = 0
    #初始化一个最佳模型，避免之后没有
    torch.save(model.state_dict(), best_model_path)

    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc, precision, recall, f1 = evaluate_model(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
            # print("best_acc is ",best_acc)
            torch.save(model.state_dict(), best_model_path)
            
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return best_acc

def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path):
    core_indices = set()
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader

    batch_gradients_cache = []
    checkpoint_cache = []
    # interval_core_loader = core_loader

    # best_model_path = args.save_best_model_path
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        # 预训练模型并缓存梯度（随机选择部分样本）
        features, labels = extract_features_and_labels(core_loader, device)

        target_core_set_size = int(len(dst_train) * args.selection_fraction)
        print(f"Target core set size: {target_core_set_size}")

        selected_indices = tmcs_shapley_selection(args,features, labels, model, target_core_set_size, device)

        core_labels = [dst_train[idx][1] for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))

        print(f"Selected {len(selected_indices)} samples for core set.")

        core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path):
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader
    
    features, labels = extract_features_and_labels(core_loader, device)

    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    print(f"Target core set size: {target_core_set_size}")

    selected_indices = tmcs_shapley_selection(args,features, labels, model, target_core_set_size, device)

    core_labels = [dst_train[idx][1] for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    print(f"Selected {len(selected_indices)} samples for core set.")

    core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

    train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path, patience=10, best_acc=0.0)

# 替换 TMCS_selection 中的逻辑
def TMCS_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, k=10):
    import logging
    from collections import Counter
    import os
    import time

    start_time = time.time()
    
    log_path = args.save_log_path + args.dataset + '/' +  args.method + '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.log'
    best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +  '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))

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
    print = lambda *args: logging.info(" ".join(map(str, args)))
    # 打印初始日志信息
    print(f"Log path: {log_path}")
    print(f"best_model_path: {best_model_path}")
    print(f"method: {args.method}")
    print(f"network: {args.model}")
    print(f"dataset: {args.dataset}")
    print(f"selection_fraction: {args.selection_fraction}")
    if args.proxy_model != 'none':
        print(f"proxy_model: {args.proxy_model}")

    print(f"数据集信息：通道数={channel}，图像大小={im_size}，类别数={num_classes}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

    # 数据集标签分布日志
    print("train set label distribution:", Counter([int(dst_train[idx][1]) for idx in range(len(dst_train))]))
    print("val set label distribution:", Counter([int(dst_val[idx][1]) for idx in range(len(dst_val))]))
    print("test set label distribution:", Counter([int(dst_test[idx][1]) for idx in range(len(dst_test))]))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # best_model_path = args.save_best_model_path

    # 预训练模型
    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # 随机选择一部分数据进行预训练
    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader

    def pretrain_model(model, data_loader, criterion, optimizer, scheduler, device, num_epochs=3):
        print("Pretraining model...")
        model.train()
        for epoch in tqdm(range(num_epochs), desc="Pretraining"):
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
        print("Pretraining completed.")

    pretrain_model(model, core_loader, criterion, optimizer, scheduler, device)

    # features, labels = extract_features_and_labels(model, random_loader, device)

    # target_core_set_size = int(len(dst_train) * args.selection_fraction)
    # print(f"Target core set size: {target_core_set_size}")

    # selected_indices = tmcs_shapley_selection(features, labels, target_core_set_size, device)

    # core_labels = [dst_train[idx][1] for idx in selected_indices]
    # print("Core set label distribution:", Counter(core_labels))

    # print(f"Selected {len(selected_indices)} samples for core set.")

    # core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

    if args.selection_framework == 'interval':
        interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path)
    elif args.selection_framework == 'once':
        once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path)


    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True),device)
    print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")
    
    os.remove(best_model_path)