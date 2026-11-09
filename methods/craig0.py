import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import nets
import torch.optim.lr_scheduler as lr_scheduler
from functorch import grad
import logging
import os
import time
from collections import Counter
from functorch import vmap, grad
from torch.utils.data import WeightedRandomSampler


def create_weighted_subset_loader(dst_train, selected_indices, batch_size=128, num_workers=1, pin_memory=True):
    # 从所选子集中获取标签
    subset_labels = [dst_train[i][1] for i in selected_indices]
    # 统计每个类别出现的次数
    class_count = np.bincount(subset_labels)
    # 根据类频次计算类权重（简单策略：某类出现越少，权重越高）
    # 避免除0错误，对于空的类别计数则不考虑即可
    class_weights = {}
    for i in range(len(class_count)):
        if class_count[i] > 0:
            class_weights[i] = 1.0 / class_count[i]
        else:
            class_weights[i] = 0.0

    # 为子集中的每个样本分配权重
    try:
        print("try")
        sample_weights = [class_weights[label] for label in subset_labels]
    except:
        print("except")
        sample_weights = [class_weights[label.item()] for label in subset_labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    # 创建WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 创建带有加权采样的DataLoader
    subset_data = Subset(dst_train, selected_indices)
    subset_loader = DataLoader(subset_data, batch_size=batch_size, 
                               num_workers=num_workers, pin_memory=pin_memory, 
                               sampler=sampler)
    return subset_loader

# def craig_select_optimized(model, data_loader, device, subset_size, args):
#     model.eval()
#     all_gradients = []
#     all_indices = []

#     criterion = nn.CrossEntropyLoss(reduction='none').to(device)

#     # 获取最后一层参数
#     last_layer_params = list(model.parameters())[-1:]

#     # 定义对单个样本计算损失的函数
#     def compute_loss_single_sample(params, input_, target_):
#         # input_: [C,H,W] (单样本), target_: 标量
#         # 我们需要扩展维度才能正常通过模型
#         input_ = input_.unsqueeze(0)  # [1,C,H,W]
#         target_ = target_.unsqueeze(0)  # [1]
#         outputs = model(input_)
#         loss = criterion(outputs, target_)  # [1]张量
#         return loss.sum()  # 返回单标量

#     # 对单样本梯度计算函数求梯度
#     grad_single = grad(compute_loss_single_sample)

#     # 使用vmap进行批处理梯度计算
#     # in_dims中: params维度为None表示不进行vmap广播，inputs和targets沿batch维度(0)映射
#     grad_batch = vmap(grad_single, in_dims=(None,0,0))

#     # 计算每个样本的最后一层梯度
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         if inputs.size(0) == 0:
#             continue

#         # 使用vmap对整批数据计算梯度
#         # grad_batch返回的梯度格式与params一致，对于最后一层只有一个参数张量，一般是 (num_classes, feature_dim)
#         # 但它会给出batch_size维度的梯度结果
#         # batch_gradients是一个元组或列表，每个元素是形状为 [batch_size, ...] 的梯度张量
#         batch_gradients = grad_batch(last_layer_params, inputs, targets)
#         # batch_gradients是针对最后一层参数列表的梯度，如只有一个参数张量时，为tuple长度1。
#         # 我们需对batch内每个样本单独flatten梯度。
#         # 注意：batch_gradients[i] 是该参数的所有样本的梯度，形状为 [batch_size, param_shape...]
        
#         # 如果last_layer_params只有一个tensor，那么：
#         # batch_gradients是类似于 (grad_tensor,) 的tuple
#         # grad_tensor形状为[batch_size, ...param dims...]
#         grad_tensor = batch_gradients[0]  
        
#         start_idx = batch_idx * data_loader.batch_size
#         for i in range(inputs.size(0)):
#             # 对单个样本的梯度flatten
#             flattened_grads = grad_tensor[i].view(-1).cpu().numpy()
#             all_gradients.append(flattened_grads)
#             all_indices.append(start_idx + i)

#     # 若没有任何梯度数据，直接返回空列表
#     if len(all_gradients) == 0:
#         print("No gradients computed")
#         return []

#     all_gradients = np.vstack(all_gradients)
#     n = len(all_gradients)
#     # print(f"Number of samples: {n}")

#     # 使用官方逻辑计算相似度矩阵（其实是距离矩阵）
#     matrix = euclidean_dist_pair_np(all_gradients)

#     # 按照官方代码进行矩阵变换
#     matrix = -1.0 * matrix
#     matrix -= (np.min(matrix) - 1e-3)

#     # 使用FacilityLocation子模函数
#     submod_function = FacilityLocation(index=np.arange(n), similarity_matrix=matrix)

#     # 使用LazyGreedy子模优化器
#     submod_optimizer = LazyGreedy(args=args, index=np.arange(n), budget=subset_size)

#     # 进行子模子集选择
#     selected_result = submod_optimizer.select(
#         gain_function=submod_function.calc_gain,
#         update_state=submod_function.update_state
#     )

#     # 将选择的索引映射回原数据集的索引
#     return [all_indices[idx] for idx in selected_result]

from sklearn.metrics.pairwise import cosine_similarity

def craig_select_optimized(model, data_loader, device, subset_size, args):
    model.eval()
    all_gradients = []
    all_indices = []

    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    # 获取最后一层参数
    last_layer_params = list(model.parameters())[-1:]

    # 定义对单个样本计算损失的函数
    def compute_loss_single_sample(params, input_, target_):
        input_ = input_.unsqueeze(0)  # [1,C,H,W]
        target_ = target_.unsqueeze(0)  # [1]
        outputs = model(input_)
        loss = criterion(outputs, target_)  # [1]张量
        return loss.sum()

    # 对单样本梯度计算函数求梯度
    grad_single = grad(compute_loss_single_sample)

    # 使用vmap进行批处理梯度计算
    grad_batch = vmap(grad_single, in_dims=(None,0,0))

    # 计算每个样本的最后一层梯度
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if inputs.size(0) == 0:
            continue

        batch_gradients = grad_batch(last_layer_params, inputs, targets)
        grad_tensor = batch_gradients[0]  

        start_idx = batch_idx * data_loader.batch_size
        for i in range(inputs.size(0)):
            flattened_grads = grad_tensor[i].view(-1).cpu().numpy()
            all_gradients.append(flattened_grads)
            all_indices.append(start_idx + i)

    if len(all_gradients) == 0:
        print("No gradients computed")
        return []

    all_gradients = np.vstack(all_gradients)
    n = len(all_gradients)

    # 使用余弦相似度
    similarity_matrix = cosine_similarity(all_gradients)
    np.fill_diagonal(similarity_matrix, 0)  # 避免自己选择自己

    # 使用FacilityLocation子模函数
    submod_function = FacilityLocation(index=np.arange(n), similarity_matrix=similarity_matrix)

    # 使用LazyGreedy子模优化器
    submod_optimizer = LazyGreedy(args=args, index=np.arange(n), budget=subset_size)

    # 进行子模子集选择
    selected_result = submod_optimizer.select(
        gain_function=submod_function.calc_gain,
        update_state=submod_function.update_state
    )

    # 将选择的索引映射回原数据集的索引
    return [all_indices[idx] for idx in selected_result]

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

def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path):
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    full_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    accumulated_core_indices = set()
    best_acc = 0.0
    try:
        num_classes = len(set(int(target.item()) for target in dst_train.targets))
        train_labels = dst_train.targets
    except:
        num_classes = len(set(int(target.item()) for target in dst_train.labels))
        train_labels = dst_train.labels
        
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        balance = True
        selected_indices = []
        core_indices = set()
        if balance:
            for c in range(num_classes):
                class_indices = np.where(np.array(train_labels) == c)[0]
                # 若该类别无样本，跳过
                if len(class_indices) == 0:
                    continue
                class_subset_size = max(1, int(len(class_indices) * args.selection_fraction))
                # 创建当前类别的数据加载器
                class_subset = Subset(dst_train, class_indices)
                class_loader = DataLoader(class_subset, batch_size=256, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
                # 使用 craig 子集选择函数
                class_selected = craig_select_optimized(model, class_loader, device, class_subset_size, args)
                # 将选择的样本索引映射回全数据集
                selected_class_indices = [class_indices[idx] for idx in class_selected]
                selected_indices.extend(selected_class_indices)
        else:
            selected_indices = craig_select_optimized(model, full_loader, device, target_core_set_size, args)

        print(f"Final number of selected samples: {len(selected_indices)}")
        # 输出select_indices的前10个
        # print(f"top 400 is {selected_indices[:400]}")

        accumulated_core_indices.update(selected_indices)
        subset_data = Subset(dst_train, selected_indices)
        subset_loader = create_weighted_subset_loader(dst_train, selected_indices, batch_size=128, num_workers=1, pin_memory=True)
        coverage_fraction = len(accumulated_core_indices) / len(dst_train)
        print(f"Coverage fraction: {coverage_fraction:.4f}")
        if best_acc > 0:
            try:
                model.load_state_dict(torch.load(best_model_path))
            except:
                pass
        best_acc = train_model(model, subset_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")

def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    subset_size = int(len(dst_train) * args.selection_fraction)
    balance = True
    selected_indices = []
    full_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # 获取数据集类别数
    try:
        num_classes = len(set(int(target.item()) for target in dst_train.targets))
        train_labels = dst_train.targets
    except:
        num_classes = len(set(int(target.item()) for target in dst_train.labels))
        train_labels = dst_train.labels

    print(f"Num classes: {num_classes}")

    if balance:
        # if hasattr(dst_train, 'labels'):
        #     train_labels = dst_train.labels
        # else:
        #     train_labels = dst_train.targets

        for c in tqdm(range(num_classes),desc="Class"):
            # 获取当前类别的样本索引
            class_indices = np.where(np.array(train_labels) == c)[0]
            # 若该类别无样本，跳过
            if len(class_indices) == 0:
                continue

            class_subset_size = max(1, int(len(class_indices) * args.selection_fraction))

            # 创建当前类别的数据加载器
            class_subset = Subset(dst_train, class_indices)
            class_loader = DataLoader(class_subset, batch_size=256, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

            print("len(class_subset_size ):",class_subset_size)
            # 使用 craig 子集选择函数
            class_selected = craig_select_optimized(model, class_loader, device, class_subset_size, args)
            # 将选择的样本索引映射回全数据集
            selected_class_indices = [class_indices[idx] for idx in class_selected]
            selected_indices.extend(selected_class_indices)
    else:
        selected_indices = craig_select_optimized(model, full_loader, device, subset_size, args)
    
    print(f"Final number of selected samples: {len(selected_indices)}")
    subset_data = Subset(dst_train, selected_indices)
    subset_loader = create_weighted_subset_loader(dst_train, selected_indices, batch_size=128, num_workers=1, pin_memory=True)
        
    train_model(model, subset_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)

def craig_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
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

    # 替换全局 print 为日志输出
    global print
    print = lambda *args: logging.info(" ".join(map(str, args)))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("CUDA is available:", torch.cuda.is_available())

    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # 根据选择框架运行相应的子集选择和训练流程
    if args.selection_framework == 'interval':
        interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    elif args.selection_framework == 'once':
        once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
    print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    os.remove(best_model_path)
    print(f"Total time: {time.time() - start_time:.4f}s")

class optimizer(object):
    def __init__(self, args, index, budget:int, already_selected=[]):
        self.args = args
        self.index = index

        if budget <= 0 or budget > index.__len__():
            raise ValueError("Illegal budget for optimizer.")

        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected

class LazyGreedy(optimizer):
    def __init__(self, args, index, budget:int, already_selected=[]):
        super(LazyGreedy, self).__init__(args, index, budget, already_selected)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            # if i % self.args.print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, self.budget))
            best_gain = -np.inf
            last_max_element = -1
            while True:
                cur_max_element = greedy_gain.argmax()
                if last_max_element == cur_max_element:
                    # Select cur_max_element into the current subset
                    selected[cur_max_element] = True
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected, **kwargs)
                    break
                new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0]
                greedy_gain[cur_max_element] = new_gain
                if new_gain >= best_gain:
                    best_gain = new_gain
                    last_max_element = cur_max_element
        return self.index[selected]


class SubmodularFunction(object):
    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        self.index = index
        self.n = len(index)

        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None

        # For the sample similarity matrix, the method supports two input modes, one is to input a pairwise similarity
        # matrix for the whole sample, and the other case allows the input of a similarity kernel to be used to
        # calculate similarities incrementally at a later time if required.
        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
            self.similarity_matrix = similarity_matrix
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

    def _similarity_kernel(self, similarity_kernel):
        return similarity_kernel


class FacilityLocation(SubmodularFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.already_selected.__len__()==0:
            self.cur_max = np.zeros(self.n, dtype=np.float32)
        else:
            self.cur_max = np.max(self.similarity_kernel(np.arange(self.n), self.already_selected), axis=1)

        self.all_idx = np.ones(self.n, dtype=bool)

    def _similarity_kernel(self, similarity_kernel):
        # Initialize a matrix to store similarity values of sample points.
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.if_columns_calculated[not_calculated] = True
            return self.sim_matrix[np.ix_(a, b)]
        return _func

    def calc_gain(self, idx_gain, selected, **kwargs):
        gains = np.maximum(0., self.similarity_kernel(self.all_idx, idx_gain) - self.cur_max.reshape(-1, 1)).sum(axis=0)
        return gains

    def calc_gain_batch(self, idx_gain, selected, **kwargs):
        batch_idx = ~self.all_idx
        batch_idx[0:kwargs["batch"]] = True
        gains = np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1, 1)).sum(axis=0)
        for i in range(kwargs["batch"], self.n, kwargs["batch"]):
            batch_idx = ~self.all_idx
            batch_idx[i * kwargs["batch"]:(i + 1) * kwargs["batch"]] = True
            gains += np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1,1)).sum(axis=0)
        return gains

    def update_state(self, new_selection, total_selected, **kwargs):
        self.cur_max = np.maximum(self.cur_max, np.max(self.similarity_kernel(self.all_idx, new_selection), axis=1))
        #self.cur_max = np.max(np.append(self.cur_max.reshape(-1, 1), self.similarity_kernel(self.all_idx, new_selection), axis=1), axis=1)

def euclidean_dist_pair_np(x):
    (rowx, colx) = x.shape
    xy = np.dot(x, x.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
    return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))