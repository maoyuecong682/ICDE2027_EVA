import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import nets

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
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return best_acc

def active_learning_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device , best_model_path):
    # 初始化核心数据集索引集合
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    b = int(target_core_set_size * args.active_fraction_ratio)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    b = max(50, b)  # 最小选择 50 个样本

    best_acc = 0.0
    trained_epochs = 0
    T = 20
    # 进行多轮挑选
    while len(core_indices) < target_core_set_size:
        print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

        model.train()  # 启用训练模式（启用 dropout）

        def shannon_entropy_function(model, X_pool, T, batch_size=256):
            outputs_list = []
            data_loader = DataLoader(X_pool, batch_size=batch_size, shuffle=False)
            
            for _ in tqdm(range(T), desc="Computing entropy"):
                model.train()  # 启用 dropout
                batch_outputs = []

                for batch in data_loader:
                    with torch.no_grad():
                        batch = batch.to(device)
                        outputs = torch.softmax(model(batch), dim=1)
                        batch_outputs.append(outputs)

                outputs_list.append(torch.cat(batch_outputs, dim=0))

            outputs_stack = torch.stack(outputs_list)
            mean_probs = outputs_stack.mean(dim=0)
            H = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6), dim=1)
            E_H = -torch.mean(torch.sum(outputs_stack * torch.log(outputs_stack + 1e-6), dim=2), dim=0)
            return H, E_H

        # 提取未选择样本的嵌入表示
        embeddings_list = []
        D_pool_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        for images, _ in tqdm(D_pool_loader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings_list.extend(images)

        X_pool = torch.stack(embeddings_list)

        H, E_H = shannon_entropy_function(model, X_pool, T)
        acquisition = H - E_H
        query_idx = torch.argsort(-acquisition)[:b]
        selected_indices = query_idx.cpu().numpy()
        core_indices.update(selected_indices)
        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")

        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

        if best_acc > 0:
            try:
                model.load_state_dict(torch.load(best_model_path))
            except:
                pass
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=1, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc)
        trained_epochs += 1

    print(f"len(core_indices): {len(core_indices)}")
    core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    train_epochs = args.epochs - trained_epochs
    train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=train_epochs, device=device, val_loader=val_loader, best_model_path=best_model_path, patience=args.patience, best_acc=best_acc)


def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    # 初始化核心数据集索引集合
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    target_core_set_size_per_selection = int(target_core_set_size * args.active_fraction_ratio)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    target_core_set_size_per_selection = target_core_set_size
    T = 100
    best_acc = 0.0
    trained_epochs = 0
    # 进行多轮挑选
    # 初始化核心数据集索引集合
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    best_acc = 0.0
    # print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

    model.train()  # 启用训练模式（启用 dropout）

    def shannon_entropy_function(model, X_pool, T, batch_size=64):
        outputs_list = []
        data_loader = DataLoader(X_pool, batch_size=batch_size, shuffle=False)
        
        for _ in tqdm(range(T), desc="Computing entropy"):
            model.train()  # 启用 dropout
            batch_outputs = []

            for batch in data_loader:
                with torch.no_grad():
                    batch = batch.to(device)
                    outputs = torch.softmax(model(batch), dim=1)
                    batch_outputs.append(outputs)

            outputs_list.append(torch.cat(batch_outputs, dim=0))

        outputs_stack = torch.stack(outputs_list)
        mean_probs = outputs_stack.mean(dim=0)
        H = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6), dim=1)
        E_H = -torch.mean(torch.sum(outputs_stack * torch.log(outputs_stack + 1e-6), dim=2), dim=0)
        return H, E_H

    # 提取未选择样本的嵌入表示
    embeddings_list = []
    D_pool_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    for images, _ in tqdm(D_pool_loader, desc="Extracting embeddings"):
        images = images.to(device)
        embeddings_list.extend(images)

    X_pool = torch.stack(embeddings_list)

    H, E_H = shannon_entropy_function(model, X_pool, T)
    acquisition = H - E_H
    query_idx = torch.argsort(-acquisition)[:target_core_set_size]
    selected_indices = query_idx.cpu().numpy()
    core_indices.update(selected_indices)
    if len(core_indices) > target_core_set_size:
        core_indices = set(list(core_indices)[:target_core_set_size])

    print(f"Selected {len(core_indices)} samples for core set.")
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in core_indices]
    print("Core set label distribution:", Counter(core_labels))

    core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

    train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, device=device, val_loader=val_loader, best_model_path=best_model_path)

    
def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion,optimizer, scheduler, device, best_model_path):
    # 初始化核心数据集索引集合
    test_loader = DataLoader(dst_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    # 初始化核心数据集索引集合，全局维护
    accumulated_core_indices = set()

    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    best_acc = 0.0
    T = 100
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection),desc="Epoch group"):
        try:
            model.load_state_dict(torch.load(best_model_path))
        except:
            pass
        # print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")
        
        core_indices = set()
        target_core_set_size = int(len(dst_train) * args.selection_fraction)
        val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        best_acc = 0.0
        # print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

        model.train()  # 启用训练模式（启用 dropout）

        def shannon_entropy_function(model, X_pool, T, batch_size=64):
            outputs_list = []
            data_loader = DataLoader(X_pool, batch_size=batch_size, shuffle=False)
            
            for _ in tqdm(range(T), desc="Computing entropy"):
                model.train()  # 启用 dropout
                batch_outputs = []

                for batch in data_loader:
                    with torch.no_grad():
                        batch = batch.to(device)
                        outputs = torch.softmax(model(batch), dim=1)
                        batch_outputs.append(outputs)

                outputs_list.append(torch.cat(batch_outputs, dim=0))

            outputs_stack = torch.stack(outputs_list)
            mean_probs = outputs_stack.mean(dim=0)
            H = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6), dim=1)
            E_H = -torch.mean(torch.sum(outputs_stack * torch.log(outputs_stack + 1e-6), dim=2), dim=0)
            return H, E_H

        # 提取未选择样本的嵌入表示
        embeddings_list = []
        D_pool_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        for images, _ in tqdm(D_pool_loader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings_list.extend(images)

        X_pool = torch.stack(embeddings_list)

        H, E_H = shannon_entropy_function(model, X_pool, T)
        acquisition = H - E_H
        query_idx = torch.argsort(-acquisition)[:target_core_set_size]
        selected_indices = query_idx.cpu().numpy()
        core_indices.update(selected_indices)
        accumulated_core_indices.update(selected_indices)
        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")

        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        if best_acc > 0:
            try:
                model.load_state_dict(torch.load(best_model_path))
            except:
                pass
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=epochs_per_selection, device=device, val_loader=val_loader, best_model_path=best_model_path,patience=epochs_per_selection,best_acc=best_acc)

    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))

    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")


def dbal_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train,dst_val, dst_test, n_query=10, T=100):
    # n_query: Number of samples to query
    # T: Number of forward passes to compute uncertainty estimates
    import logging
    from collections import Counter
    import os
    import time
    start_time = time.time()
    
    # log_path = args.save_log_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target) + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
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

    # 替换全局 print
    global print
    print = lambda *args: logging.info(" ".join(map(str, args)))
    print("###############################Start  Selection...###############################")
    print("")
    # 打印初始日志信息
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

    # 数据集标签分布日志
    print("train set label distribution:", Counter([int(dst_train[idx][1]) for idx in range(len(dst_train))]))
    print("val set label distribution:", Counter([int(dst_val[idx][1]) for idx in range(len(dst_val))]))
    print("test set label distribution:", Counter([int(dst_test[idx][1]) for idx in range(len(dst_test))]))
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 初始化模型
    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    
    def pretrained_model(model, data_loader, criterion, optimizer, num_epochs):
        for epoch in tqdm(range(num_epochs), desc="Training pretrain model"):
            model.train()
            running_loss = 0.0
            for inputs, labels_batch in data_loader:
                # print(f"Pretrain - Input shape: {inputs.shape}")  # 调试信息
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

    random_size = int(len(dst_train) * args.selection_fraction)
    ramndom_indices = np.random.choice(len(dst_train), random_size, replace=False)
    pretrained_model(model, DataLoader(Subset(dst_train, ramndom_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True), criterion, optimizer, 3)
    # pretrained_model(model, DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=1, pin_memory=True), criterion, optimizer, 1)

    if args.selection_framework == 'once':
        once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    elif args.selection_framework == 'active_learning':
        active_learning_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    elif args.selection_framework == 'interval':
        interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)


    # 使用 BALD 方法进行数据挑选
    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True),device=device)
    print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    os.remove(best_model_path)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")