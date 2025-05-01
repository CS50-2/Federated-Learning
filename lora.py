import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
import time


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=8.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha / rank

        m, n = self.linear.weight.shape
        self.A = nn.Parameter(torch.randn(m, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(n, rank))

    def forward(self, x):
        delta_W = self.alpha * (self.A @ self.B.T)
        return self.linear(x) + F.linear(x, delta_W)

    def set_requires_grad(self, lora_only=True):
        self.linear.requires_grad_(not lora_only)
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)


class MLPModel(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=200, num_classes=10, use_lora=False, rank=8, lora_alpha=1.0):
        super(MLPModel, self).__init__()
        self.use_lora = use_lora

        fc1 = nn.Linear(input_size, hidden_size)
        fc2 = nn.Linear(hidden_size, hidden_size)
        fc3 = nn.Linear(hidden_size, num_classes)

        if use_lora:
            self.fc1 = LoRALinear(fc1, rank=rank, alpha=lora_alpha)
            self.fc2 = LoRALinear(fc2, rank=rank, alpha=lora_alpha)
            self.fc3 = LoRALinear(fc3, rank=rank, alpha=lora_alpha)
        else:
            self.fc1 = fc1
            self.fc2 = fc2
            self.fc3 = fc3

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_requires_grad(self, lora_only=True):
        for module in self.children():
            if hasattr(module, 'set_requires_grad'):
                module.set_requires_grad(lora_only)


# 加载 MNIST 数据集
def load_mnist_data(data_path="./data"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    if os.path.exists(os.path.join(data_path, "MNIST/raw/train-images-idx3-ubyte")):
        print("✅ MNIST 数据集已存在，跳过下载。")
    else:
        print("⬇️ 正在下载 MNIST 数据集...")

    train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)

    # visualize_mnist_samples(train_data)
    return train_data, test_data


def split_data_by_label(dataset, num_clients=10):
    """
    手动划分数据集，每个客户端包含 10 个类别，并自定义样本数量。
    :param dataset: 原始数据集（如 MNIST）
    :param num_clients: 客户端总数
    :return: (客户端数据集, 客户端数据大小)
    """
    # 手动划分的样本数量（每个客户端 10 个类别的数据量）
    client_data_sizes = {
        0: {0: 600},
        1: {1: 700},
        2: {2: 500},
        3: {3: 600},
        4: {4: 600},
        5: {5: 500},
        6: {6: 500},
        7: {7: 500},
        8: {8: 500},
        9: {9: 500}
    }

    # 统计每个类别的数据索引
    label_to_indices = {i: [] for i in range(10)}  # 记录每个类别的索引
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    # 初始化客户端数据存储
    client_data_subsets = {}
    client_actual_sizes = {i: {label: 0 for label in range(10)} for i in range(num_clients)}  # 记录实际分配的数据量

    # 遍历每个客户端，为其分配指定类别的数据
    for client_id, label_info in client_data_sizes.items():
        selected_indices = []  # 临时存储该客户端所有选中的索引
        for label, size in label_info.items():
            # 确保不超出类别数据集实际大小
            available_size = len(label_to_indices[label])
            sample_size = min(available_size, size)

            if sample_size < size:
                print(f"⚠️ 警告：类别 {label} 的数据不足，客户端 {client_id} 只能获取 {sample_size} 条样本（需求 {size} 条）")

            # 从该类别中随机抽取样本
            sampled_indices = random.sample(label_to_indices[label], sample_size)
            selected_indices.extend(sampled_indices)

            # 记录实际分配的数据量
            client_actual_sizes[client_id][label] = sample_size

        # 创建 PyTorch Subset
        client_data_subsets[client_id] = torch.utils.data.Subset(dataset, selected_indices)

    # 打印每个客户端的实际分配数据量
    print("\n📊 每个客户端实际数据分布:")
    for client_id, label_sizes in client_actual_sizes.items():
        print(f"客户端 {client_id}: {label_sizes}")

    return client_data_subsets, client_actual_sizes


# 本地训练函数
def local_train(model, train_loader, epochs=5, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model.state_dict()


#  联邦平均聚合函数
def fed_avg(global_model, client_state_dicts, client_sizes):
    global_dict = global_model.state_dict()
    subkey = [sublist[0] for sublist in client_state_dicts]
    new_client_sizes = dict(([(key, client_sizes[key]) for key in subkey]))
    total_data = sum(sum(label_sizes.values()) for label_sizes in new_client_sizes.values())  # 计算所有客户端数据总量
    for key in global_dict.keys():
        global_dict[key] = sum(
            client_state[key] * (sum(new_client_sizes[client_id].values()) / total_data)
            for (client_id, client_state) in client_state_dicts
        )
    global_model.load_state_dict(global_dict)
    return global_model


# 评估模型
def evaluate(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total * 100
    return total_loss / len(test_loader), accuracy

def entropy_weight(l):
    entropies = []
    for X in l:
        P = X / (np.sum(X) + 1e-12)  # 归一化得到概率分布
        K = 1 / np.log(len(X))
        E = -K * np.sum(P * np.log(P + 1e-12))  # 计算熵，越大越无区分度
        entropies.append(E)

        # 算信息量
    information_gain = [1 - e for e in entropies]
    # 归一化
    sum_ig = sum(information_gain)
    weights = [ig / sum_ig for ig in information_gain]

    return weights


def calculate_GRC(global_model, client_models, client_losses):
    """
    正确计算 GRC 分数，并修正后续步骤
    """

    # 1. 计算客户端指标（参数差异）
    param_diffs = []
    for model in client_models:
        diff = 0.0
        for g_param, l_param in zip(global_model.parameters(), model.parameters()):
            diff += torch.norm(g_param - l_param).item()
        param_diffs.append(diff)

    # 2. 对 losses 和 diffs 进行正确 mapping
    def map_sequence_loss(sequence):
        max_val = max(sequence)
        min_val = min(sequence)
        return [(max_val - x) / (max_val + min_val) for x in sequence]  # 【✔】负相关

    def map_sequence_diff(sequence):
        max_val = max(sequence)
        min_val = min(sequence)
        return [(x - min_val) / (max_val + min_val) for x in sequence]  # 【✔】正相关

    client_losses = map_sequence_loss(client_losses)
    param_diffs = map_sequence_diff(param_diffs)

    # 3. 构建参考序列 (理想值 = 1)
    ref_loss = 1.0
    ref_diff = 1.0

    # 4. 计算每个指标的 Δ
    all_deltas = []
    for loss, diff in zip(client_losses, param_diffs):
        all_deltas.append(abs(loss - ref_loss))
        all_deltas.append(abs(diff - ref_diff))
    max_delta = max(all_deltas)
    min_delta = min(all_deltas)

    # 5. 计算灰色关联系数 (GRC)，ρ=0.5
    grc_losses = []
    grc_diffs = []
    for loss, diff in zip(client_losses, param_diffs):
        delta_loss = abs(loss - ref_loss)
        delta_diff = abs(diff - ref_diff)

        grc_loss = (min_delta + 0.5 * max_delta) / (delta_loss + 0.5 * max_delta)
        grc_diff = (min_delta + 0.5 * max_delta) / (delta_diff + 0.5 * max_delta)

        grc_losses.append(grc_loss)
        grc_diffs.append(grc_diff)

    grc_losses = np.array(grc_losses)
    grc_diffs = np.array(grc_diffs)

    # 6. 计算熵权（基于原始mapped数据）
    grc_metrics = np.vstack([client_losses, param_diffs])  # 【注意】这里 shape 是 (2, n_clients)
    weights = entropy_weight(grc_metrics)  # 【✔】熵权算的是原mapped指标，不是grc！

    # 7. 加权求和，注意是【乘法】不是除法
    weighted_score = grc_losses * weights[0] + grc_diffs * weights[1]  # 【修改点】乘法！

    return weighted_score, weights

def get_out_features(layer):
    if isinstance(layer, LoRALinear):
        return layer.linear.out_features
    else:
        return layer.out_features

def calculate_flops(model, input_size):
    flops = 0

    # 兼容输入为 (batch_size, 784) 或 (batch_size, 1, 28, 28)
    if len(input_size) == 2:
        in_features = input_size[1]
    elif len(input_size) == 3:
        in_features = input_size[1] * input_size[2]
    elif len(input_size) == 4:
        in_features = input_size[1] * input_size[2] * input_size[3]
    else:
        raise ValueError("Unsupported input shape for FLOPs calculation")

    fc1_out = get_out_features(model.fc1)
    fc2_out = get_out_features(model.fc2)
    fc3_out = get_out_features(model.fc3)

    # FLOPs = 2 * input_dim * output_dim (multiply + add)
    flops += 2 * in_features * fc1_out
    flops += 2 * fc1_out * fc2_out
    flops += 2 * fc2_out * fc3_out

    return flops



# Function to calculate TFLOPs for one forward pass
def calculate_tflops(model, inputs):
    # Calculate FLOPs
    flops = calculate_flops(model, input_size=inputs.shape)
    
    # Measure time for a single forward pass
    start_time = time.time()
    outputs = model(inputs)
    end_time = time.time()
    
    # Time taken for the forward pass (in seconds)
    elapsed_time = end_time - start_time
    
    # Calculate TFLOPs (total FLOPs divided by time in seconds, divided by 10^12 to get TFLOPs)
    tflops = flops / (elapsed_time * 1e12)
    
    return tflops

def select_clients(client_loaders, use_all_clients=False, num_select=None,
                   select_by_loss=False, global_model=None, grc=False, is_lora=False):
    if grc:  # 使用 GRC 选择客户端
        client_models = []
        # 1. 训练本地模型并计算损失
        client_losses = []
        for client_id, client_loader in client_loaders.items():
            # If using LoRA, update the model type
            if is_lora:
                local_model = MLPModel(use_lora=True) # Use LoRA model
            else: 
                local_model = MLPModel()  # Default to MLP model
            local_model = load_model_weights(local_model, global_model, is_lora)  # Load weights into the model
            local_train(local_model, client_loader, epochs=1, lr=0.01)
            client_models.append(local_model)
            loss, _ = evaluate(local_model, client_loader)
            client_losses.append(loss)

        # 2. 计算 GRC 分数
        grc_scores, grc_weights = calculate_GRC(global_model, client_models, client_losses)
        select_clients.latest_weights = grc_weights  # 记录权重

        # 3. 按 GRC 分数排序（从高到低，GRC越高表示越好）
        client_grc_pairs = list(zip(client_loaders.keys(), grc_scores))
        client_grc_pairs.sort(key=lambda x: x[1], reverse=True)  # 降序排序

        # 4. 选择 GRC 最高的前 num_select 个客户端
        selected = [client_id for client_id, _ in client_grc_pairs[:num_select]]

        return selected

    if use_all_clients is True:
        print("Selecting all clients")
        return list(client_loaders.keys())

    if num_select is None:
        raise ValueError("If use_all_clients=False, num_select cannot be None!")

    if select_by_loss and global_model:
        client_losses = {}
        for client_id, loader in client_loaders.items():
            local_model = MLPModel()
            local_model.load_state_dict(global_model.state_dict())

            local_train(local_model, loader, epochs=1, lr=0.01)

            loss, _ = evaluate(local_model, loader)
            client_losses[client_id] = loss
        selected_clients = sorted(client_losses, key=client_losses.get, reverse=True)[:num_select]
        print(f"Selected {num_select} clients with the highest loss: {selected_clients}")
    else:
        selected_clients = random.sample(list(client_loaders.keys()), num_select)
        print(f"Randomly selected {num_select} clients: {selected_clients}")

    return selected_clients

def update_communication_counts(communication_counts, selected_clients, event):
    """
    客户端通信计数
    - event='receive' 表示客户端接收到全局模型
    - event='send' 表示客户端上传本地模型
    - event='full_round' 仅在客户端完成完整收发时增加
    """
    for client_id in selected_clients:
        communication_counts[client_id][event] += 1

        # 仅当客户端完成一次完整的 send 和 receive 时增加 full_round
        if event == "send" and communication_counts[client_id]['receive'] > 0:
            communication_counts[client_id]['full_round'] += 1

def perform_local_training(selected_clients, client_loaders, global_model, client_data_sizes, communication_counts, is_lora=False):
    client_state_dicts = []
    round_tflops_list = []
    for client_id in selected_clients:
        client_loader = client_loaders[client_id]

        local_model = MLPModel(use_lora=is_lora)

        # 加载全局模型权重
        local_model = load_model_weights(local_model, global_model, is_lora)

        # 本地训练
        local_state = local_train(local_model, client_loader, epochs=5, lr=0.01)
        client_state_dicts.append((client_id, local_state))
        update_communication_counts(communication_counts, [client_id], "send")

        param_mean = {name: param.mean().item() for name, param in local_model.named_parameters()}
        print(f"  ✅ 客户端 {client_id} 训练完成 | 样本数量: {sum(client_data_sizes[client_id].values())}")
        print(f"  📌 客户端 {client_id} 模型参数均值: {param_mean}")

    return client_state_dicts, round_tflops_list


def load_model_weights(local_model, global_model, is_lora=False):
    """ 
    Load weights from the global model to the local model. 
    If `is_lora` is True, ignore LoRA-specific parameters (A and B).
    """
    model_state_dict = global_model.state_dict()

    # If it's the LoRA model, filter out the LoRA-specific parameters (A and B matrices)
    if is_lora:
        filtered_state_dict = {k: v for k, v in model_state_dict.items() if 'A' not in k and 'B' not in k}
        local_model.load_state_dict(filtered_state_dict, strict=False)  # Load the filtered state dict
    else:
        local_model.load_state_dict(model_state_dict)  # For MLP model, load all parameters
    
    return local_model


def run_experiment(grc, rounds, client_loaders, client_data_sizes, test_loader, is_lora=False, csv_path=None):
    model = MLPModel(use_lora=is_lora)
    communication_counts = {client_id: {'send': 0, 'receive': 0, 'full_round': 0} for client_id in client_loaders}

    # define if it is LoRA or not 
    prefix = "LoRA" if is_lora else "Original"

    all_results = []

    for r in range(rounds):
        print(f"\n🔄 第 {r + 1} 轮 {'LoRA' if is_lora else 'Original'}")

        selected_clients = select_clients(
            client_loaders, use_all_clients=False, num_select=2,
            select_by_loss=False, global_model=model, grc=grc, is_lora=is_lora
        )

        update_communication_counts(communication_counts, selected_clients, "receive")

        client_state_dicts, _ = perform_local_training(
            selected_clients, client_loaders, model,
            client_data_sizes, communication_counts, is_lora=is_lora
        )

        model = fed_avg(model, client_state_dicts, client_data_sizes)
        loss, acc = evaluate(model, test_loader)
        tflops = calculate_tflops(model, torch.randn(32, 28 * 28))
        total_comm = sum(communication_counts[c]['send'] + communication_counts[c]['receive'] for c in selected_clients)

        row = {
            "Round": r + 1,
            f"{prefix}_Accuracy": acc,
            f"{prefix}_Loss": loss,
            f"{prefix}_TFLOPs": tflops,
            f"{prefix}_Comm": total_comm,
            f"{prefix}_SelectedClients": selected_clients
        }

        all_results.append(row)

    # --- Updating CSV ---
    df_new = pd.DataFrame(all_results).set_index("Round")

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path).set_index("Round")
        df_combined = df_existing.combine_first(df_new) if not is_lora else df_existing.combine(df_new, lambda s1, s2: s1 if s1.notnull().all() else s2)
    else:
        df_combined = df_new

    df_combined.reset_index().to_csv(csv_path, index=False)


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    train_data, test_data = load_mnist_data()
    client_datasets, client_data_sizes = split_data_by_label(train_data)
    client_loaders = {cid: data.DataLoader(ds, batch_size=32, shuffle=True) for cid, ds in client_datasets.items()}
    test_loader = data.DataLoader(test_data, batch_size=32)

    # Initializing filename 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"combined_results_{timestamp}.csv"

    rounds = 200

    # MLP
    run_experiment(
        grc=False,
        rounds=rounds,
        client_loaders=client_loaders,
        client_data_sizes=client_data_sizes,
        test_loader=test_loader,
        is_lora=False,
        csv_path=csv_filename
    )

    # LoRA
    run_experiment(
        grc=False,
        rounds=rounds,
        client_loaders=client_loaders,
        client_data_sizes=client_data_sizes,
        test_loader=test_loader,
        is_lora=True,
        csv_path=csv_filename
    )

    print(f"✅ 所有模型训练完成，结果保存至：{csv_filename}")



if __name__ == "__main__":
    main()
