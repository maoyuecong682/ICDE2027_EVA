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
    train_loader = DataLoader(dst_train, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    b = max(50, b)  # 最小选择 50 个样本
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    best_acc = 0.0
    trained_epochs = 0
    # 进行多轮挑选
    while len(core_indices) < target_core_set_size:
        print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

        # 提取未选择样本的嵌入表示（批量处理）
        embeddings_list = []
        D_pool_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)

        for data_feature, _ in tqdm(D_pool_loader, desc="Extracting embeddings"):
            data_feature = data_feature.to(device)
            with torch.no_grad():
                outputs = model.feature_extractor(data_feature)  # 使用特征提取器
                outputs = outputs.view(outputs.size(0), -1)  # 展平
            embeddings = outputs.detach().cpu().numpy()
            embeddings_list.extend(embeddings)

        # 转换为单一 numpy 数组，保持二维结构
        embeddings_array = np.stack(embeddings_list, axis=0)  # [num_samples, embedding_dim]

        # 转换为 Tensor
        embeddings_tensor = torch.tensor(embeddings_array, device=device)  # [num_samples, embedding_dim]

        # 这里假设你要将嵌入向量传递给全连接层进行分类

        with torch.no_grad():
            logits = model.fc(embeddings_tensor)  # 直接使用全连接层
            probs = torch.nn.functional.softmax(logits, dim=1)

        # 排除已选择的样本
        all_indices = np.arange(len(embeddings_tensor))
        remaining_indices = np.setdiff1d(all_indices, list(core_indices))

        uncertainty_type =  args.uncertainty_type
        if uncertainty_type == "BvSB":
            # 计算 Best-versus-Second Best (BvSB) 分数
            top_two = torch.topk(probs, 2, dim=1).values
            bv_sb_scores = (top_two[:, 0] - top_two[:, 1]).cpu().numpy()
            selected_scores = bv_sb_scores[remaining_indices]
            selected_indices = remaining_indices[np.argsort(-selected_scores)[:b]]
        elif uncertainty_type == "max_entropy":
            # 计算 Entropy (EP) 分数
            entropy_scores = (-torch.sum(probs * torch.log(probs + 1e-6), dim=1)).cpu().numpy()
            selected_scores = entropy_scores[remaining_indices]
            selected_indices = remaining_indices[np.argsort(-selected_scores)[:b]]
        elif uncertainty_type == "max_variation_ratio":
            # 使用MC Dropout计算 Variation Ratios
            scores, labels, idx_list = [], [], []
            
            # 确保模型处于训练模式以启用Dropout（如有必要）
            # model.train()

            with torch.no_grad():
                for i, (images, targets) in enumerate(tqdm(train_loader, desc="Calculating Uncertainty")):
                    images = images.to(device)
                    mc_outputs = []

                    for _ in range(100):
                        # 确保在MC时模型带有dropout
                        # 可以在外部设置 model.train() 或者通过指定 dropout 层激活
                        out = torch.nn.functional.softmax(model(images), dim=1)
                        mc_outputs.append(out.cpu().numpy())

                    mc_outputs = np.array(mc_outputs)  # (T, batch_size, num_classes)
                    mean_probs = np.mean(mc_outputs, axis=0)  # (batch_size, num_classes)
                    max_probs = np.max(mean_probs, axis=1)
                    variation_ratios = 1 - max_probs

                    scores.extend(variation_ratios)
                    labels.extend(targets.numpy())
                    # 假设这里的索引方式对应数据集顺序（无shuffle情形）
                    start_idx = i * 128
                    end_idx = min((i + 1) * 128, len(dst_train))
                    idx_list.extend(range(start_idx, end_idx))

            scores = np.array(scores)
            idx_list = np.array(idx_list)

            # 只选择尚未被选择过的样本
            mask = np.isin(idx_list, remaining_indices)
            selected_scores = scores[mask]
            # 更新 remaining_indices，使得与 selected_scores 一一对应
            filtered_indices = idx_list[mask]
            # 选择样本
            selected_indices = filtered_indices[np.argsort(-selected_scores)[:b]]

        elif uncertainty_type == "mean_std":
            # 使用MC Dropout计算Mean STD
            scores, labels, idx_list = [], [], []

            # 确保模型在训练模式启用dropout（如有需要）
            # model.train()

            with torch.no_grad():
                for i, (images, targets) in enumerate(tqdm(train_loader, desc="Calculating Uncertainty")):
                    images = images.to(device)
                    mc_outputs = []
                    for _ in range(100):
                        out = torch.nn.functional.softmax(model(images), dim=1)
                        mc_outputs.append(out.cpu().numpy())

                    mc_outputs = np.array(mc_outputs)  # (T, batch_size, num_classes)
                    stds = np.std(mc_outputs, axis=0)  # (batch_size, num_classes)
                    mean_stds = np.mean(stds, axis=1)  # (batch_size,)

                    scores.extend(mean_stds)
                    labels.extend(targets.numpy())
                    start_idx = i * 128
                    end_idx = min((i + 1) * 128, len(dst_train))
                    idx_list.extend(range(start_idx, end_idx))

            scores = np.array(scores)
            idx_list = np.array(idx_list)

            # 只选择尚未被选择过的样本
            mask = np.isin(idx_list, remaining_indices)
            selected_scores = scores[mask]
            # 过滤indices
            filtered_indices = idx_list[mask]
            selected_indices = filtered_indices[np.argsort(-selected_scores)[:b]]


        # selected_indices = remaining_indices[np.argsort(-selected_scores)[:b]]
        core_indices.update(selected_indices)

        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")

        # 在核心数据集上继续训练模型
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
    best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=train_epochs, device=device, val_loader=val_loader, best_model_path=best_model_path,patience=args.patience, best_acc=best_acc)

def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    # 初始化核心数据集索引集合
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    b = int(target_core_set_size * args.active_fraction_ratio)
    b = max(50, b)  # 最小选择 50 个样本

    train_loader = DataLoader(dst_train, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    best_acc = 0.0
    trained_epochs = 0

    print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

    # 提取未选择样本的嵌入表示（批量处理）
    D_pool_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    embeddings_list = []
    for data_feature, _ in tqdm(D_pool_loader, desc="Extracting embeddings"):
        data_feature = data_feature.to(device)
        with torch.no_grad():
            outputs = model.feature_extractor(data_feature)  # 使用特征提取器
            outputs = outputs.view(outputs.size(0), -1)  # 展平
        embeddings = outputs.detach().cpu().numpy()
        embeddings_list.extend(embeddings)

    embeddings_array = np.stack(embeddings_list, axis=0)  # [num_samples, embedding_dim]
    embeddings_tensor = torch.tensor(embeddings_array, device=device)  # [num_samples, embedding_dim]

    # 直接使用全连接层获取logits
    with torch.no_grad():
        logits = model.fc(embeddings_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)

    all_indices = np.arange(len(embeddings_tensor))
    remaining_indices = np.setdiff1d(all_indices, list(core_indices))

    uncertainty_type =  args.uncertainty_type

    if uncertainty_type == "BvSB":
        # 计算 BvSB 分数
        top_two = torch.topk(probs, 2, dim=1).values
        bv_sb_scores = (top_two[:, 0] - top_two[:, 1]).cpu().numpy()
        selected_scores = bv_sb_scores[remaining_indices]

    elif uncertainty_type == "max_entropy":
        # 计算熵
        entropy_scores = (-torch.sum(probs * torch.log(probs + 1e-6), dim=1)).cpu().numpy()
        selected_scores = entropy_scores[remaining_indices]

    elif uncertainty_type == "max_variation_ratio":
        # 使用MC Dropout计算 Variation Ratios
        scores, labels, idx_list = [], [], []
        
        # 确保模型处于训练模式以启用Dropout（如有必要）
        # model.train()

        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(train_loader, desc="Calculating Uncertainty")):
                images = images.to(device)
                mc_outputs = []

                for _ in range(100):
                    # 确保在MC时模型带有dropout
                    # 可以在外部设置 model.train() 或者通过指定 dropout 层激活
                    out = torch.nn.functional.softmax(model(images), dim=1)
                    mc_outputs.append(out.cpu().numpy())

                mc_outputs = np.array(mc_outputs)  # (T, batch_size, num_classes)
                mean_probs = np.mean(mc_outputs, axis=0)  # (batch_size, num_classes)
                max_probs = np.max(mean_probs, axis=1)
                variation_ratios = 1 - max_probs

                scores.extend(variation_ratios)
                labels.extend(targets.numpy())
                # 假设这里的索引方式对应数据集顺序（无shuffle情形）
                start_idx = i * 128
                end_idx = min((i + 1) * 128, len(dst_train))
                idx_list.extend(range(start_idx, end_idx))

        scores = np.array(scores)
        idx_list = np.array(idx_list)

        # 只选择尚未被选择过的样本
        mask = np.isin(idx_list, remaining_indices)
        selected_scores = scores[mask]
        # 更新 remaining_indices，使得与 selected_scores 一一对应
        filtered_indices = idx_list[mask]
        # 选择样本
        selected_indices = filtered_indices[np.argsort(-selected_scores)[:target_core_set_size]]

        core_indices.update(selected_indices)
        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")

        # 后续训练和返回
        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc)
        return

    elif uncertainty_type == "mean_std":
        # 使用MC Dropout计算Mean STD
        scores, labels, idx_list = [], [], []

        # 确保模型在训练模式启用dropout（如有需要）
        # model.train()

        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(train_loader, desc="Calculating Uncertainty")):
                images = images.to(device)
                mc_outputs = []
                for _ in range(100):
                    out = torch.nn.functional.softmax(model(images), dim=1)
                    mc_outputs.append(out.cpu().numpy())

                mc_outputs = np.array(mc_outputs)  # (T, batch_size, num_classes)
                stds = np.std(mc_outputs, axis=0)  # (batch_size, num_classes)
                mean_stds = np.mean(stds, axis=1)  # (batch_size,)

                scores.extend(mean_stds)
                labels.extend(targets.numpy())
                start_idx = i * 128
                end_idx = min((i + 1) * 128, len(dst_train))
                idx_list.extend(range(start_idx, end_idx))

        scores = np.array(scores)
        idx_list = np.array(idx_list)

        # 只选择尚未被选择过的样本
        mask = np.isin(idx_list, remaining_indices)
        selected_scores = scores[mask]
        # 过滤indices
        filtered_indices = idx_list[mask]
        selected_indices = filtered_indices[np.argsort(-selected_scores)[:target_core_set_size]]

        core_indices.update(selected_indices)
        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")

        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc)
        return

    # 如果不是 max_variation_ratio 或 mean_std，直接使用 selected_scores 和 remaining_indices 进行选择
    selected_indices = remaining_indices[np.argsort(-selected_scores)[:target_core_set_size]]
    core_indices.update(selected_indices)

    if len(core_indices) > target_core_set_size:
        core_indices = set(list(core_indices)[:target_core_set_size])
        
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in core_indices]
    print("Core set label distribution:", Counter(core_labels))

    print(f"Selected {len(core_indices)} samples for core set.")

    # 在核心数据集上继续训练模型
    core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc)

def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion,optimizer, scheduler, device, best_model_path):
    # 初始化核心数据集索引集合
    test_loader = DataLoader(dst_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    b = int(target_core_set_size * args.active_fraction_ratio)
    train_loader = DataLoader(dst_train, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    b = target_core_set_size
    # 初始化核心数据集索引集合，全局维护
    accumulated_core_indices = set()

    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection),desc="Epoch group"):
        core_indices = set()
        try:
            model.load_state_dict(torch.load(best_model_path))
        except:
            pass
        print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")
        # 提取未选择样本的嵌入表示（批量处理）
        embeddings_list = []
        D_pool_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)

        for data_feature, _ in tqdm(D_pool_loader, desc="Extracting embeddings"):
            data_feature = data_feature.to(device)
            with torch.no_grad():
                outputs = model.feature_extractor(data_feature)  # 使用特征提取器
                outputs = outputs.view(outputs.size(0), -1)  # 展平
            embeddings = outputs.detach().cpu().numpy()
            embeddings_list.extend(embeddings)

        # 转换为单一 numpy 数组，保持二维结构
        embeddings_array = np.stack(embeddings_list, axis=0)  # [num_samples, embedding_dim]

        # 转换为 Tensor
        embeddings_tensor = torch.tensor(embeddings_array, device=device)  # [num_samples, embedding_dim]

        # 这里假设你要将嵌入向量传递给全连接层进行分类

        with torch.no_grad():
            logits = model.fc(embeddings_tensor)  # 直接使用全连接层
            probs = torch.nn.functional.softmax(logits, dim=1)

        # 排除已选择的样本
        all_indices = np.arange(len(embeddings_tensor))
        remaining_indices = np.setdiff1d(all_indices, list(core_indices))

        uncertainty_type =  args.uncertainty_type
        if uncertainty_type == "BvSB":
            # 计算 Best-versus-Second Best (BvSB) 分数
            top_two = torch.topk(probs, 2, dim=1).values
            bv_sb_scores = (top_two[:, 0] - top_two[:, 1]).cpu().numpy()
            selected_scores = bv_sb_scores[remaining_indices]
            selected_indices = remaining_indices[np.argsort(-selected_scores)[:b]]
        elif uncertainty_type == "max_entropy":
            # 计算 Entropy (EP) 分数
            entropy_scores = (-torch.sum(probs * torch.log(probs + 1e-6), dim=1)).cpu().numpy()
            selected_scores = entropy_scores[remaining_indices]
            selected_indices = remaining_indices[np.argsort(-selected_scores)[:b]]
        elif uncertainty_type == "max_variation_ratio":
            # 使用MC Dropout计算 Variation Ratios
            scores, labels, idx_list = [], [], []
            
            # 确保模型处于训练模式以启用Dropout（如有必要）
            # model.train()

            with torch.no_grad():
                for i, (images, targets) in enumerate(tqdm(train_loader, desc="Calculating Uncertainty")):
                    images = images.to(device)
                    mc_outputs = []

                    for _ in range(100):
                        # 确保在MC时模型带有dropout
                        # 可以在外部设置 model.train() 或者通过指定 dropout 层激活
                        out = torch.nn.functional.softmax(model(images), dim=1)
                        mc_outputs.append(out.cpu().numpy())

                    mc_outputs = np.array(mc_outputs)  # (T, batch_size, num_classes)
                    mean_probs = np.mean(mc_outputs, axis=0)  # (batch_size, num_classes)
                    max_probs = np.max(mean_probs, axis=1)
                    variation_ratios = 1 - max_probs

                    scores.extend(variation_ratios)
                    labels.extend(targets.numpy())
                    # 假设这里的索引方式对应数据集顺序（无shuffle情形）
                    start_idx = i * 128
                    end_idx = min((i + 1) * 128, len(dst_train))
                    idx_list.extend(range(start_idx, end_idx))

            scores = np.array(scores)
            idx_list = np.array(idx_list)

            # 只选择尚未被选择过的样本
            mask = np.isin(idx_list, remaining_indices)
            selected_scores = scores[mask]
            # 更新 remaining_indices，使得与 selected_scores 一一对应
            filtered_indices = idx_list[mask]
            # 选择样本
            selected_indices = filtered_indices[np.argsort(-selected_scores)[:b]]

        elif uncertainty_type == "mean_std":
            # 使用MC Dropout计算Mean STD
            scores, labels, idx_list = [], [], []

            # 确保模型在训练模式启用dropout（如有需要）
            # model.train()

            with torch.no_grad():
                for i, (images, targets) in enumerate(tqdm(train_loader, desc="Calculating Uncertainty")):
                    images = images.to(device)
                    mc_outputs = []
                    for _ in range(100):
                        out = torch.nn.functional.softmax(model(images), dim=1)
                        mc_outputs.append(out.cpu().numpy())

                    mc_outputs = np.array(mc_outputs)  # (T, batch_size, num_classes)
                    stds = np.std(mc_outputs, axis=0)  # (batch_size, num_classes)
                    mean_stds = np.mean(stds, axis=1)  # (batch_size,)

                    scores.extend(mean_stds)
                    labels.extend(targets.numpy())
                    start_idx = i * 128
                    end_idx = min((i + 1) * 128, len(dst_train))
                    idx_list.extend(range(start_idx, end_idx))

            scores = np.array(scores)
            idx_list = np.array(idx_list)

            # 只选择尚未被选择过的样本
            mask = np.isin(idx_list, remaining_indices)
            selected_scores = scores[mask]
            # 过滤indices
            filtered_indices = idx_list[mask]
            selected_indices = filtered_indices[np.argsort(-selected_scores)[:b]]


        # selected_indices = remaining_indices[np.argsort(-selected_scores)[:b]]
        core_indices.update(selected_indices)

        accumulated_core_indices.update(selected_indices)

        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")

        # 在核心数据集上继续训练模型
        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        if best_acc > 0:
                try:
                    model.load_state_dict(torch.load(best_model_path))
                except:
                    pass
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=epochs_per_selection, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc,patience=epochs_per_selection)
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")

def mcal_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train,dst_val, dst_test, k=10):
    import logging
    from collections import Counter
    import os

    import time
    start_time = time.time()
    
    # log_path = args.save_log_path + args.dataset + '/' +  args.method + '/' + args.uncertainty_type + '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method + '/' + args.uncertainty_type + '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
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
    # 打印初始日志信息
    print(f"Log path: {log_path}")
    print(f"method: {args.method}")
    print(f"selection_framework: {args.selection_framework}")
    print(f"selection_interval: {args.selection_interval}")
    print(f"uncertainty_type: {args.uncertainty_type}")
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

    # 初始化模型（只初始化一次）
    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    # best_model_path = args.save_best_model_path

    def pretrained_model(model, data_loader, criterion, optimizer, num_epochs):
        for epoch in tqdm(range(num_epochs), desc="Training pretrain model"):
            model.train()
            running_loss = 0.0
            for inputs, labels_batch in data_loader:
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

    # pretrained_model(model, DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=1, pin_memory=True), criterion, optimizer, 5)
    random_size = int(len(dst_train) * args.selection_fraction)
    ramndom_indices = np.random.choice(len(dst_train), random_size, replace=False)
    pretrained_model(model, DataLoader(Subset(dst_train, ramndom_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True), criterion, optimizer, 3)
    
    if args.selection_framework == 'active_learning':
        active_learning_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    elif args.selection_framework == 'once':
        once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    elif args.selection_framework == 'interval':
        interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)

    # 最终评估模型
    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
    print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

    # 删除模型的pth文件
    os.remove(best_model_path)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")

