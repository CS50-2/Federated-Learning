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

# 定义 MLP 模型
class MLPModel(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=200, num_classes=10):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compute_training_flops(model, frozen_layers=None):
    """
    计算整个模型一次训练(iter)的 FLOPs（前向+反向+SGD更新），
    frozen_layers 列表中的层只计算前向 FLOPs。
    返回 (total_flops, per_layer_dict)
    """
    if frozen_layers is None:
        frozen_layers = []

    total_flops = 0
    flops_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            out_f = module.out_features

            # 1) 前向
            fwd = 2 * in_f * out_f + out_f

            if name in frozen_layers:
                layer_flops = fwd
            else:
                # 2) 反向
                bw_w    = 2 * in_f * out_f      # 权重梯度
                bw_in   =     in_f * out_f      # 输入梯度
                bw_b    =         out_f         # 偏置梯度

                # 3) SGD 更新
                upd_w   = 2 * in_f * out_f      # 权重更新
                upd_b   = 2 * out_f             # 偏置更新

                layer_flops = fwd + (bw_w + bw_in + bw_b) + (upd_w + upd_b)

            flops_dict[name] = layer_flops
            total_flops += layer_flops

    return total_flops, flops_dict


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


def select_clients(client_loaders, use_all_clients=False, num_select=None,
                   select_by_loss=False, global_model=None, grc=False):
    if grc:  # 使用 GRC 选择客户端
        client_models = []
        # 1. 训练本地模型并计算损失
        client_losses = []
        for client_id, client_loader in client_loaders.items():
            local_model = MLPModel()
            local_model.load_state_dict(global_model.state_dict())  # 同步全局模型
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

    # 其余选择逻辑保持不变

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

def perform_local_training(selected_clients, client_loaders, global_model, client_data_sizes, communication_counts):
    client_state_dicts = []
    round_tflops_list = []
    for client_id in selected_clients:
        client_loader = client_loaders[client_id]
        local_model = MLPModel()
        local_model.load_state_dict(global_model.state_dict())
        
        # 本地训练
        local_state = local_train(local_model, client_loader, epochs=1, lr=0.1)
        client_state_dicts.append((client_id, local_state))
        update_communication_counts(communication_counts, [client_id], "send")
 
        param_mean = {name: param.mean().item() for name, param in local_model.named_parameters()}
        print(f"  ✅ 客户端 {client_id} 训练完成 | 样本数量: {sum(client_data_sizes[client_id].values())}")
        print(f"  📌 客户端 {client_id} 模型参数均值: {param_mean}")


    return client_state_dicts, round_tflops_list

def run_experiment(grc, rounds, client_loaders, client_data_sizes, test_loader):
    global_model = MLPModel()
    results = []
    communication_counts = {client_id: {'send': 0, 'receive': 0, 'full_round': 0} for client_id in client_loaders.keys()}
    

    for r in range(rounds):
        print(f"\n🔄 实验 grc={grc}）第 {r + 1} 轮")
        selected_clients = select_clients(client_loaders, use_all_clients=False, num_select=2,
                                          select_by_loss=False, global_model=global_model, grc=grc)
        print(f"  选中客户端: {selected_clients}")
        update_communication_counts(communication_counts, selected_clients, "receive")
        client_state_dicts, round_tflops_list = perform_local_training(
            selected_clients, client_loaders, global_model,
            client_data_sizes, communication_counts
        )
        total_send = sum(communication_counts[c]['send'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
        total_receive = sum(communication_counts[c]['receive'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
        total_comm = total_send + total_receive
        if results:
            total_comm += results[-1]["TotalComm"]
        global_model = fed_avg(global_model, client_state_dicts, client_data_sizes)
        loss, accuracy = evaluate(global_model, test_loader)
        avg_tflops = np.mean(round_tflops_list) if round_tflops_list else 0
        result_dict = {
            "Round": r+1,
            "Accuracy": accuracy,
            "TotalComm": total_comm,
            "AvgTFlops": avg_tflops
        }
        results.append(result_dict)
        print(f"  轮 {r+1} 测试准确率: {accuracy:.2f}%，平均 t-FLOPs: {avg_tflops:.0f}")
    return results

def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # 加载数据集、划分数据与构建 DataLoader
    train_data, test_data = load_mnist_data()
    client_datasets, client_data_sizes = split_data_by_label(train_data)
    client_loaders = {client_id: data.DataLoader(dataset, batch_size=32, shuffle=True)
                      for client_id, dataset in client_datasets.items()}
    test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    rounds = 200  # 联邦学习轮数
    
    # setting experiments 
    experiment_1 = run_experiment(
        grc=True,
        rounds=rounds,
        client_loaders=client_loaders,
        client_data_sizes=client_data_sizes,
        test_loader=test_loader
    )
    
    # # Combine two results into one 
    # combined_results = []
    # for r in range(rounds):
    #     row = {
    #         "Round": exp_no_freeze[r]["Round"],
    #         "Accuracy_NoFreeze": exp_no_freeze[r]["Accuracy"],
    #         "TotalComm_NoFreeze": exp_no_freeze[r]["TotalComm"],
    #         "Accuracy_Freeze": exp_freeze[r]["Accuracy"],
    #         "TotalComm_Freeze": exp_freeze[r]["TotalComm"],
    #     }
    #     combined_results.append(row)
    
    df = pd.DataFrame(experiment_1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"combined_training_data_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n所有实验结果已保存到 {csv_filename}")

    # # 绘制测试准确率随轮次变化对比图
    # plt.figure(figsize=(10, 6))
    # plt.plot(df["Round"], df["Accuracy_NoFreeze"], marker='o', label="No Freeze")
    # # plt.plot(df["Round"], df["Accuracy_Freeze"], marker='s', label="Freeze (APF on fc2)")
    # plt.xlabel("Federated Learning Rounds")
    # plt.ylabel("Test Accuracy")
    # plt.title("Test Accuracy vs Rounds")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()