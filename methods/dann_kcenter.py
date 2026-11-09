import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from nets.DANN import DANN
from methods.methods_utils import euclidean_dist
from torchvision import transforms
from copy import deepcopy

from collections import Counter
import os
import time
import logging


######################################################
###  2. 下面是 k_center_greedy 和辅助函数，保持不变    ###
######################################################
def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None, print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # 随机选择一个初始点
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # 初始化距离矩阵
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]


class EmbeddingExtractor_DANN(nn.Module):
    """
    适用于 DANN 的特征提取器，只调用 model.feature_extractor。
    如果你的下游只需要中间特征，可使用这个类来替代旧的 EmbeddingExtractor。
    """
    def __init__(self, dann_model):
        super(EmbeddingExtractor_DANN, self).__init__()
        self.model = dann_model

    def forward(self, x):
        # x: (batch_size, channel, H, W)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        # 只调用特征提取器，不做标签预测/域分类
        features = self.model.feature_extractor(x)
        return features


######################################################
###  3. 下面是 kcentergreedy_selection 主流程示例      ###
###      核心是用 DANN 替代原模型，并做相应修改        ###
######################################################
def dann_kcenter_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, balance=False):

    start_time = time.time()    

    # 日志和文件夹准备
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'
    best_model_path = args.save_best_model_path + args.dataset + '/' +  str(args.selection_fraction)  +'/' + args.method + '.pth'
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

    # 替换全局 print 为 logging
    global print
    print = lambda *x: logging.info(" ".join(map(str, x)))

    # 打印一些信息
    print(f"Log path: {log_path}")
    print(f"method: {args.method}")
    print(f"dataset: {args.dataset}")
    print(f"selection_fraction: {args.selection_fraction}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    if torch.cuda.is_available():
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

    print(f"数据集信息：通道数={channel}，图像大小={im_size}，类别数={num_classes}")

    print("train set label distribution:", Counter([int(dst_train[idx][1]) for idx in range(len(dst_train))]))
    print("val set label distribution:", Counter([int(dst_val[idx][1]) for idx in range(len(dst_val))]))
    print("test set label distribution:", Counter([int(dst_test[idx][1]) for idx in range(len(dst_test))]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ######################################################
    ### 修改开始：使用 DANN 模型(仅用于分类) 并做特征提取 ###
    ######################################################
    use_extractor = False
    try:
        # 如果你有其他预训练 DANN 的方式，这里可以自行扩展
        # 比如加载预训练权重之类，但演示中直接随机初始化
        model_dann = DANN(channel=channel, 
                          num_classes=num_classes, 
                          im_size=im_size, 
                          alpha=1.0).to(device)
        model_dann.eval()

        # 用 EmbeddingExtractor_DANN 来抽取中间特征
        extractor = EmbeddingExtractor_DANN(model_dann).to(device)
        extractor.eval()

        train_loader = DataLoader(dst_train, batch_size=128, shuffle=False)
        data, labels = [], []
        with torch.no_grad():
            for images, targets in tqdm(train_loader, desc="Extracting embeddings"):
                images = images.to(device)
                embeddings = extractor(images)  # 仅调用 feature_extractor
                data.append(embeddings.cpu().numpy())
                labels.append(targets.numpy())

        data = np.concatenate(data)
        labels = np.concatenate(labels)
        use_extractor = True

    except Exception as e:
        # 如果上面过程出错(例如某些数据集或模型不兼容)，则直接展平输入使用
        print("Extractor failed, use plain flatten. Error msg:", e)
        train_loader = DataLoader(dst_train, batch_size=128, shuffle=False)
        data, labels = [], []
        for images, targets in tqdm(train_loader, desc="Building dataset"):
            # 直接展平
            data.append(images.view(images.size(0), -1).numpy())
            labels.append(targets.numpy())
        data = np.concatenate(data)
        labels = np.concatenate(labels)
    ######################################################
    ### 修改结束                                         ###
    ######################################################

    # 参数设置
    n = data.shape[0]
    core_set_size = int(n * args.selection_fraction)
    print(f"Core set size: {core_set_size}")

    # 如果需要平衡选样
    if balance:
        print("Performing balanced k-Center Greedy selection per class.")
        selected_indices = []
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            class_data = torch.tensor(data[class_indices], dtype=torch.float32, device=device)
            class_core_set_size = max(1, int(core_set_size * (len(class_indices) / n)))  # 至少选 1 个

            selected_class_indices = k_center_greedy(
                class_data,
                budget=class_core_set_size,
                metric=euclidean_dist,
                device=device,
                random_seed=args.random_seed,
                index=class_indices,
                already_selected=[]
            )
            selected_indices.extend(selected_class_indices)

        selected_indices = np.array(selected_indices)
    else:
        print("Performing global k-Center Greedy selection.")
        matrix = torch.tensor(data, dtype=torch.float32, device=device)
        selected_indices = k_center_greedy(
            matrix,
            budget=core_set_size,
            metric=euclidean_dist,
            device=device,
            random_seed=args.random_seed,
            already_selected=[]
        )

    print(f"Selected {len(selected_indices)} samples.")
    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    # 创建核心集数据加载器
    subset_data = Subset(dst_train, selected_indices)
    core_loader = DataLoader(subset_data, batch_size=128, shuffle=True)

    # 如果用了中间特征提取器，这里可以释放它和临时模型
    if use_extractor:
        del extractor
        del model_dann
        torch.cuda.empty_cache()

    ######################################################
    ### 修改开始：重新实例化一个可训练的 DANN 用于分类   ###
    ######################################################
    train_model_instance = DANN(channel=channel, 
                                num_classes=num_classes, 
                                im_size=im_size, 
                                alpha=1.0).to(device)
    train_model_instance.train()
    ######################################################
    ### 修改结束                                         ###
    ######################################################

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(train_model_instance.parameters(), 
                          lr=args.selection_lr, 
                          momentum=args.selection_momentum, 
                          weight_decay=args.selection_weight_decay, 
                          nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate_model(model, data_loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels_batch in data_loader:
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                ### DANN 做标签预测时传 domain=False
                outputs = model(inputs, domain=False)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return acc, precision, recall, f1

    def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, patience=10):
        best_acc = 0.0
        trigger_times = 0

        for epoch in tqdm(range(num_epochs), desc="Training model"):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels_batch) in enumerate(data_loader):
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                ### DANN 做标签预测时传 domain=False
                outputs = model(inputs, domain=False)
                loss = criterion(outputs, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()  # 更新学习率

            # 每个 epoch 结束后验证
            acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_val, batch_size=128, shuffle=False))
            if acc > best_acc:
                best_acc = acc
                trigger_times = 0
                # 保存最佳模型
                torch.save(model.state_dict(), best_model_path)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping triggered.")
                    break

    # 训练模型
    train_model(train_model_instance, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, patience=args.patience)

    # 测试并输出结果
    train_model_instance.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(train_model_instance, DataLoader(dst_test, batch_size=128, shuffle=False))
    print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

    # 删除最佳模型文件
    os.remove(best_model_path)  
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")