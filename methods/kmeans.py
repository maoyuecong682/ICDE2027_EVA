import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # 用于进度条显示
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import mydatasets
import nets
from sklearn.cluster import MiniBatchKMeans
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---- 批大小（EVA 追加）-----------------------------------------------------
# 见 methods/random.py 同名常量处的说明。
DATA_BATCH_SIZE = 128



class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.model = model
        self.model.fc = nn.Identity()  # 替换最后的全连接层

    def forward(self, x):
        x = self.model(x)
        return x

# 定义验证函数
def evaluate_model(model, data_loader, device, task_type = "classification", return_details=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)

            if task_type == 'classification':
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
            elif task_type == 'regression':
                preds = outputs.reshape(-1)  # Also valid for a final batch of one.
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.reshape(-1).cpu().numpy())

    if task_type == 'classification':
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        if return_details:
            return dict(accuracy=acc,precision=precision,recall=recall,f1=f1,
                precision_macro=precision_score(all_labels,all_preds,average='macro',zero_division=0),
                recall_macro=recall_score(all_labels,all_preds,average='macro',zero_division=0),
                f1_macro=f1_score(all_labels,all_preds,average='macro',zero_division=0))
        return acc, precision, recall, f1
    elif task_type == 'regression':
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        return mse, mae

def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, device, best_model_path, task_type="classification", val_loader=None, patience=10):
    best_metric = float('inf') if task_type == 'regression' else -float('inf')
    trigger_times = 0
    epochs_completed = 0

    for epoch in tqdm(range(num_epochs), desc="Training model"):
        epochs_completed = epoch + 1
        model.train()
        running_loss = 0.0

        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)

            if task_type == 'classification':
                loss = criterion(outputs, labels_batch)
            elif task_type == 'regression':
                loss = criterion(outputs.reshape(-1), labels_batch.float().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()  # 更新学习率

        # 每个 epoch 结束后验证模型
        if val_loader:
            if task_type == 'classification':
                acc, precision, recall, f1 = evaluate_model(model, val_loader, device, task_type)
                metric = acc  # 使用准确率作为分类任务的判断标准
            elif task_type == 'regression':
                mse, mae = evaluate_model(model, val_loader, device, task_type)
                metric = mse  # 使用 MSE 作为回归任务的判断标准

            # 早停机制
            if (task_type == 'classification' and metric > best_metric) or \
               (task_type == 'regression' and metric < best_metric):
                best_metric = metric
                trigger_times = 0
                # 保存最佳模型权重
                torch.save(model.state_dict(), best_model_path)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping triggered.")
                    break
    return dict(epochs_completed=epochs_completed,best_validation_metric=float(best_metric),
                stopped_early=epochs_completed < num_epochs)

def kmeans_selection(args,channel,im_size,num_classes,class_names,mean,std, dst_train,dst_val,dst_test):

    import logging
    from collections import Counter
    import os
    import time
    start_time = time.time()
    
    # log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  str(args.selection_fraction)  +'/' + args.method + '.pth'
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' +  str(args.selection_fraction)  +'/' + args.method + '.log'
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

    # 替换全局 print
    global print
    print = lambda *args: logging.info(" ".join(map(str, args)))
    # 打印初始日志信息
    try:
        print(f"Log path: {log_path}")
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

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print("CUDA is available:", torch.cuda.is_available())
        # best_model_path = args.save_best_model_path + args.dataset + '_' +  args.method +'_' + str(args.selection_fraction) + '_' + str(args.balance_target)  + '.pth'
        task_type = args.task_type
        print(f"Task type: {task_type}")
        if task_type == 'classification':
            print("train set label distribution:", Counter([int(dst_train[idx][1]) for idx in range(len(dst_train))]))
            print("val set label distribution:", Counter([int(dst_val[idx][1]) for idx in range(len(dst_val))]))
            print("test set label distribution:", Counter([int(dst_test[idx][1]) for idx in range(len(dst_test))]))


        # 初始化模型用于提取嵌入
        global bool_del
        bool_del = 0

        select_start = time.time()

        # 初始化模型并提取嵌入
        try:
            model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
            extractor = EmbeddingExtractor(model).to(device)
            extractor.eval()

            # 使用提取器获取数据嵌入
            train_loader = DataLoader(dst_train, batch_size=DATA_BATCH_SIZE, shuffle=False)
            data, labels = [], []
            with torch.no_grad():
                for images, targets in tqdm(train_loader, desc="Extracting embeddings"):
                    images = images.to(device)
                    embeddings = extractor(images)
                    data.append(embeddings.cpu().numpy())
                    labels.append(targets.numpy())
            data = np.concatenate(data)
            labels = np.concatenate(labels)
            bool_del = 1
            
        except:
            # 创建数据加载器
            train_loader = DataLoader(dst_train, batch_size=DATA_BATCH_SIZE, shuffle=False)
            data, labels = [], []

            # 收集所有数据点和标签
            for images, targets in tqdm(train_loader, desc="Building dataset"):
                data.append(images.view(images.size(0), -1).numpy())
                labels.append(targets.numpy())
            data = np.concatenate(data)
            labels = np.concatenate(labels)

        # 参数设置
        if task_type == 'classification':
            k = num_classes  # 可调整 k 值，默认为数据的类别数
        elif task_type == 'regression':
            k = 10
        epsilon = 0.1  # 误差容忍度
        d = data.shape[1]  # 数据的维度
        d = min(d, 20)  # 限制维度，防止数值溢出
        n = data.shape[0]  # 数据集大小

        # 核心集大小计算
        core_set_size = int(n * args.selection_fraction)  # 使用 args.selection_fraction 控制核心集比例
        print(f"Core set size: {core_set_size}")

        # 使用 MiniBatchKMeans 加速
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096)

        # 可以用 tqdm 包装 fit 方法，显示进度条
        for _ in tqdm(range(10), desc="Clustering"):
            kmeans.partial_fit(data)

        # 获取聚类中心和标签
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        # 构建核心集
        representative_indices = []
        # for cluster in tqdm(range(k)):
        #     cluster_points = np.where(cluster_labels == cluster)[0]
        #     distances = np.linalg.norm(data[cluster_points] - cluster_centers[cluster], axis=1)
        #     # 按距离排序并选择最远的数据点
        #     # sorted_indices = np.argsort(distances)
        #     sorted_indices = np.argsort(distances)[::-1]
        #     num_representatives = min(len(cluster_points), core_set_size // k)
        #     selected_points = cluster_points[sorted_indices[:num_representatives]]
        #     representative_indices.extend(selected_points)
        for cluster in tqdm(range(k)):
            cluster_points = np.where(cluster_labels == cluster)[0]
            print(f"Cluster {cluster}: {len(cluster_points)} points")
            distances = np.linalg.norm(data[cluster_points] - cluster_centers[cluster], axis=1)
            sorted_indices = np.argsort(distances)[::-1]
            num_representatives = max(1, min(len(cluster_points), core_set_size // k))
            print(f"Selecting {num_representatives} representatives from cluster {cluster}")
            selected_points = cluster_points[sorted_indices[:num_representatives]]
            representative_indices.extend(selected_points)
        
        if task_type == 'classification':
            from collections import Counter
            core_labels = [int(dst_train[idx][1]) for idx in representative_indices]
            print("Core set label distribution:", Counter(core_labels))


        # 创建核心集数据加载器
        subset = Subset(dst_train, representative_indices)
        
        # 所有核心数据为dst_train+subset
        all_core_data = subset
        core_loader = DataLoader(all_core_data, batch_size=DATA_BATCH_SIZE, shuffle=True)

        select_end = time.time()

        # 分离用于训练的模型
        if bool_del == 1:
            del model
            del extractor
            torch.cuda.empty_cache()


        # 初始化新的模型用于训练
        train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
        train_model_instance.train()

        # 定义损失函数、优化器和学习率调度器
        optimizer = optim.SGD(train_model_instance.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # 初始化损失函数
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss().to(device)
        elif task_type == "regression":
            criterion = nn.MSELoss().to(device)  # 使用均方误差损失函数

        # 训练模型
        train_model(
            model=train_model_instance,
            data_loader=core_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.epochs,
            device=device,
            best_model_path=best_model_path,
            task_type=task_type,
            val_loader=DataLoader(dst_val, batch_size=DATA_BATCH_SIZE, shuffle=False),
            patience=args.patience
        )
        
        # 加载最佳模型并评估
        train_model_instance.load_state_dict(torch.load(best_model_path))
        if task_type == 'classification':
            acc, precision, recall, f1 = evaluate_model(train_model_instance, DataLoader(dst_test, batch_size=DATA_BATCH_SIZE, shuffle=False), device, task_type)
            print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        elif task_type == 'regression':
            mse, mae = evaluate_model(train_model_instance, DataLoader(dst_test, batch_size=DATA_BATCH_SIZE, shuffle=False), device, task_type)
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
        os.remove(best_model_path)
        end_time = time.time()
        print(f"Selection time elapsed: {select_end - select_start:.2f} seconds.")
        print(f"Time cost: {end_time - start_time:.2f}s")
    except Exception as e:
        # 记录异常信息到日志
        logging.error(f"An error occurred in Arnoldi_selection: {str(e)}")
        logging.error("Stack trace: ", exc_info=True)  # 记录完整的错误堆栈信息
