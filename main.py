
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import numpy as np
# import random
# import os
# import matplotlib.pyplot as plt
# import csv
# import pandas as pd
# from datetime import datetime
#
#
# # 定义 MLP 模型
# class MLPModel(nn.Module):
#     def __init__(self):
#         super(MLPModel, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 200)  # 第一层，输入维度 784 -> 200
#         self.fc2 = nn.Linear(200, 200)  # 第二层，200 -> 200
#         self.fc3 = nn.Linear(200, 10)  # 输出层，200 -> 10
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # 展平输入 (batch_size, 1, 28, 28) -> (batch_size, 784)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # 直接输出，不使用 Softmax（因为 PyTorch 的 CrossEntropyLoss 里已经包含了）
#         return x
#
#
# # 加载 MNIST 数据集
# def load_mnist_data(data_path="./data"):
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#
#     if os.path.exists(os.path.join(data_path, "MNIST/raw/train-images-idx3-ubyte")):
#         print("✅ MNIST 数据集已存在，跳过下载。")
#     else:
#         print("⬇️ 正在下载 MNIST 数据集...")
#
#     train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
#     test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
#
#     # visualize_mnist_samples(train_data)
#     return train_data, test_data
#
#
# # 显示数据集示例图片
# def visualize_mnist_samples(dataset, num_samples=10):
#     fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.2, 1.5))
#     for i in range(num_samples):
#         img, label = dataset[i]
#         axes[i].imshow(img.squeeze(), cmap="gray")
#         axes[i].set_title(label)
#         axes[i].axis("off")
#     plt.show()
#
#
# def split_data_by_label(dataset, num_clients=10):
#
#     client_data_sizes = {
#         0: {0: 600},
#         1: {1: 700},
#         2: {2: 500},
#         3: {3: 600},
#         4: {4: 600},
#         5: {5: 500},
#         6: {6: 100},
#         7: {7: 100},
#         8: {8: 100},
#         9: {9: 100}
#     }
#
#     # 统计每个类别的数据索引
#     label_to_indices = {i: [] for i in range(10)}  # 记录每个类别的索引
#     for idx, (_, label) in enumerate(dataset):
#         label_to_indices[label].append(idx)
#
#     # 初始化客户端数据存储
#     client_data_subsets = {}
#     client_actual_sizes = {i: {label: 0 for label in range(10)} for i in range(num_clients)}  # 记录实际分配的数据量
#
#     # 遍历每个客户端，为其分配指定类别的数据
#     for client_id, label_info in client_data_sizes.items():
#         selected_indices = []  # 临时存储该客户端所有选中的索引
#         for label, size in label_info.items():
#             # 确保不超出类别数据集实际大小
#             available_size = len(label_to_indices[label])
#             sample_size = min(available_size, size)
#
#             if sample_size < size:
#                 print(f"⚠️ 警告：类别 {label} 的数据不足，客户端 {client_id} 只能获取 {sample_size} 条样本（需求 {size} 条）")
#
#             # 从该类别中随机抽取样本
#             sampled_indices = random.sample(label_to_indices[label], sample_size)
#             selected_indices.extend(sampled_indices)
#
#             # 记录实际分配的数据量
#             client_actual_sizes[client_id][label] = sample_size
#
#         # 创建 PyTorch Subset
#         client_data_subsets[client_id] = torch.utils.data.Subset(dataset, selected_indices)
#
#     # 打印每个客户端的实际分配数据量
#     print("\n📊 每个客户端实际数据分布:")
#     for client_id, label_sizes in client_actual_sizes.items():
#         print(f"客户端 {client_id}: {label_sizes}")
#
#     return client_data_subsets, client_actual_sizes
#
#
# # 本地训练函数
# def local_train(model, train_loader, epochs=5, lr=0.01):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     model.train()
#     for epoch in range(epochs):
#         for batch_x, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#     return model.state_dict()
#
#
# # 实时记录训练过程中的loss用做Fedgra
# def local_train_fedgra_loss(model, train_loader, epochs=5, lr=0.01):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     model.train()
#     loss_sq_sum = 0.0
#
#     for epoch in range(epochs):
#         for batch_x, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             loss_sq_sum += loss.item() ** 2
#
#     h_i = loss_sq_sum  # 放大训练过程中 loss 的累积程度，从而增强客户端之间的区分度。
#     return model, h_i
#
#
# #  联邦平均聚合函数
# def fed_avg(global_model, client_state_dicts, client_sizes):
#     global_dict = global_model.state_dict()
#     subkey = [sublist[0] for sublist in client_state_dicts]
#     new_client_sizes = dict(([(key, client_sizes[key]) for key in subkey]))
#     total_data = sum(sum(label_sizes.values()) for label_sizes in new_client_sizes.values())  # 计算所有客户端数据总量
#     for key in global_dict.keys():
#         global_dict[key] = sum(
#             client_state[key] * (sum(new_client_sizes[client_id].values()) / total_data)
#             for (client_id, client_state) in client_state_dicts
#         )
#     global_model.load_state_dict(global_dict)
#     return global_model
#
#
# # 评估模型
# def evaluate(model, test_loader):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     correct, total, total_loss = 0, 0, 0.0
#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += batch_y.size(0)
#             correct += (predicted == batch_y).sum().item()
#     accuracy = correct / total * 100
#     return total_loss / len(test_loader), accuracy
#
#
# # 熵权法实现
#
# def entropy_weight(l):
#     l = np.array(l)
#
#     # Step 1: Min-Max 归一化（避免负值和爆炸）
#     X_norm = (l - l.min(axis=1, keepdims=True)) / (l.max(axis=1, keepdims=True) - l.min(axis=1, keepdims=True) + 1e-12)
#
#     # Step 2: 转为概率矩阵 P_ki
#     P = X_norm / (X_norm.sum(axis=1, keepdims=True) + 1e-12)
#
#     # Step 3: 计算熵
#     K = 1 / np.log(X_norm.shape[1])
#     E = -K * np.sum(P * np.log(P + 1e-12), axis=1)  # shape: (2,)
#
#     # Step 4: 计算信息效用值 & 权重
#     d = 1 - E
#     weights = d / np.sum(d)
#     return weights.tolist()
#
#
# # 灰色关联度实现
# def calculate_GRC(global_model, client_models, client_losses):
#     """
#     计算 GRC 分数 + 熵权法权重
#     修正：
#       - 映射顺序错误
#       - 熵权法使用错误指标
#     """
#     # 正确写法：使用整体参数向量计算 L2 范数（符合文献）
#     param_diffs = []
#     for model in client_models:
#         global_vec = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
#         local_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
#         diff = torch.norm(local_vec - global_vec).item()
#         param_diffs.append(diff)
#
#     # 2. 映射原始指标到 [0, 1] 区间（为熵权法准备）
#     def map_sequence_loss(sequence):
#         max_val, min_val = max(sequence), min(sequence)
#         denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
#         return [(max_val - x) / denom for x in sequence]  # 越小越好，负相关
#
#     def map_sequence_diff(sequence):
#         max_val, min_val = max(sequence), min(sequence)
#         denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
#         return [(x - min_val) / denom for x in sequence]  # 越大越好，正相关
#
#     # 用于 GRC 的映射
#     mapped_losses = map_sequence_loss(client_losses)
#     mapped_diffs = map_sequence_diff(param_diffs)
#
#     # 3. 熵权法计算权重（根据映射值，非 GRC）
#     grc_metrics = np.vstack([mapped_losses, mapped_diffs])  # shape: (2, n_clients)
#     weights = entropy_weight(grc_metrics)  # w_loss, w_diff
#
#     # 4. 计算 GRC 分数（ξki），参考值为 1
#     ref_loss, ref_diff = 1.0, 1.0
#     delta_losses = [abs(x - ref_loss) for x in mapped_losses]
#     delta_diffs = [abs(x - ref_diff) for x in mapped_diffs]
#     all_deltas = delta_losses + delta_diffs
#     max_delta, min_delta = max(all_deltas), min(all_deltas)
#
#     grc_losses = []
#     grc_diffs = []
#     rho = 0.5  # 区分度因子
#     for d_loss, d_diff in zip(delta_losses, delta_diffs):
#         grc_loss = (min_delta + rho * max_delta) / (d_loss + rho * max_delta)
#         grc_diff = (min_delta + rho * max_delta) / (d_diff + rho * max_delta)
#         grc_losses.append(grc_loss)
#         grc_diffs.append(grc_diff)
#
#     # 5. 加权求和得到最终 GRC 分数
#     grc_losses = np.array(grc_losses)
#     grc_diffs = np.array(grc_diffs)
#     weighted_score = grc_losses * weights[0] + grc_diffs * weights[1]
#
#     # 调试（每个客户端的 loss、diff、得分）
#     print("\n GRC得分]")
#     for i in range(len(client_models)):
#         print(f"Client {i} | loss: {client_losses[i]:.4f}, diff: {param_diffs[i]:.4f}, "
#               f"mapped_loss: {mapped_losses[i]:.4f}, mapped_diff: {mapped_diffs[i]:.4f}, "
#               f"GRC: {weighted_score[i]:.4f}")
#     print(f"熵权法权重: w_loss = {weights[0]:.4f}, w_diff = {weights[1]:.4f}")
#
#     return weighted_score, weights
#
#
# # 客户端选择器
# def select_clients(client_loaders, use_all_clients=False, num_select=None,
#                    select_by_loss=False, global_model=None, grc=False,
#                    fairness_tracker=None):
#     if grc:  # 使用 GRC 选择客户端
#         client_models = []
#         # 1. 训练本地模型并计算损失
#         client_losses = []
#         for client_id, client_loader in client_loaders.items():
#             local_model = MLPModel()
#             local_model.load_state_dict(global_model.state_dict())  # 同步全局模型
#             trained_model, h_i = local_train_fedgra_loss(local_model, client_loader, epochs=5, lr=0.01)
#             client_models.append(trained_model)
#             client_losses.append(h_i)
#
#         # 2. 计算 GRC 分数
#         grc_scores, grc_weights = calculate_GRC(global_model, client_models, client_losses)
#         select_clients.latest_weights = grc_weights  # 记录权重
#
#         # 3. 按 GRC 分数排序（从高到低，GRC越高表示越好）
#         client_grc_pairs = list(zip(client_loaders.keys(), grc_scores))
#         client_grc_pairs.sort(key=lambda x: x[1], reverse=True)  # 降序排序
#
#         # 4. 选择 GRC 最高的前 num_select 个客户端
#         selected_clients = [client_id for client_id, _ in client_grc_pairs[:num_select]]
#         return selected_clients
#
#     # 其余选择逻辑保持不变
#
#     if use_all_clients is True:
#         print("Selecting all clients")
#         return list(client_loaders.keys())
#
#     if num_select is None:
#         raise ValueError("If use_all_clients=False, num_select cannot be None!")
#
#     if select_by_loss and global_model:
#         client_losses = {}
#
#         for client_id, loader in client_loaders.items():
#             local_model = MLPModel()
#             local_model.load_state_dict(global_model.state_dict())
#             local_train(local_model, loader, epochs=5, lr=0.01)
#             loss, _ = evaluate(local_model, loader)
#             client_losses[client_id] = loss
#         selected_clients = sorted(client_losses, key=client_losses.get, reverse=True)[:num_select]
#         print(f"Selected {num_select} clients with the highest loss: {selected_clients}")
#     else:
#         selected_clients = random.sample(list(client_loaders.keys()), num_select)
#         print(f"Randomly selected {num_select} clients: {selected_clients}")
#
#     return selected_clients
#
#
# def update_communication_counts(communication_counts, selected_clients, event):
#
#     for client_id in selected_clients:
#         communication_counts[client_id][event] += 1
#
#
#         if event == "send" and communication_counts[client_id]['receive'] > 0:
#             communication_counts[client_id]['full_round'] += 1
#
#
# def main():
#     T1 = time.time()
#     torch.manual_seed(0)
#     random.seed(0)
#     np.random.seed(0)
#
#     # 加载 MNIST 数据集
#     train_data, test_data = load_mnist_data()
#
#     # 生成客户端数据集，每个客户端包含多个类别
#     client_datasets, client_data_sizes = split_data_by_label(train_data)
#
#     # 创建数据加载器
#     client_loaders = {client_id: data.DataLoader(dataset, batch_size=32, shuffle=True)
#                       for client_id, dataset in client_datasets.items()}
#     test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)
#
#     # 初始化全局模型
#     global_model = MLPModel()
#     global_accuracies = []  # 记录每轮全局模型的测试集准确率
#     total_communication_counts = []  # 记录每轮客户端通信次数
#     rounds = 300  # 联邦学习轮数
#     use_all_clients = False  # 是否进行客户端选择
#     num_selected_clients = 2  # 每轮选择客户端训练数量
#     use_loss_based_selection = False  # 是否根据 loss 选择客户端
#     grc = True
#
#     # 初始化通信计数器
#     communication_counts = {}
#     for client_id in client_loaders.keys():
#         communication_counts[client_id] = {
#             'send': 0,
#             'receive': 0,
#             'full_round': 0
#         }
#
#     # 实验数据存储 CSV
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv_filename = f"training_data_{timestamp}.csv"
#     csv_data = []
#
#     for r in range(rounds):
#         print(f"\n🔄 第 {r + 1} 轮聚合")
#         # 选择客户端
#
#         selected_clients = select_clients(
#             client_loaders,
#             use_all_clients=use_all_clients,
#             num_select=num_selected_clients,
#             select_by_loss=use_loss_based_selection,
#             global_model=global_model,
#             grc=grc
#         )
#         # # 设置随机阻断某个客户端的接收记录（验证用）
#         # blocked_client = random.choice(selected_clients)
#         # print(f" Blocking client {blocked_client} from receiving, skipping the receive event record.")
#
#         # for client_id in selected_clients:
#         #     if client_id == blocked_client:
#         #         continue  # 直接跳过 receive 记录
#         #     update_communication_counts(communication_counts, [client_id], "receive")
#
#         # 记录客户端接收通信次数
#         update_communication_counts(communication_counts, selected_clients, "receive")
#         client_state_dicts = []
#
#         # 客户端本地训练
#         for client_id in selected_clients:
#             client_loader = client_loaders[client_id]
#             local_model = MLPModel()
#             local_model.load_state_dict(global_model.state_dict())  # 复制全局模型参数
#             local_state = local_train(local_model, client_loader, epochs=5, lr=0.01)  # 训练 1 轮
#             client_state_dicts.append((client_id, local_state))  # 存储 (客户端ID, 训练后的参数)
#
#             update_communication_counts(communication_counts, [client_id], "send")  # 记录客户端上报通信次数
#
#             param_mean = {name: param.mean().item() for name, param in local_model.named_parameters()}
#             print(f"  ✅ 客户端 {client_id} 训练完成 | 样本数量: {sum(client_data_sizes[client_id].values())}")
#             print(f"  📌 客户端 {client_id} 模型参数均值: {param_mean}")
#
#         # 计算本轮通信次数
#         total_send = sum(
#             communication_counts[c]['send'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
#         total_receive = sum(
#             communication_counts[c]['receive'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
#         total_comm = total_send + total_receive  # 每轮独立的总通信次数
#
#         # 如果不是第一轮，累加前一轮的通信次数
#         if len(total_communication_counts) > 0:
#             total_comm += total_communication_counts[-1]
#         total_communication_counts.append(total_comm)
#
#         # 聚合模型参数
#         global_model = fed_avg(global_model, client_state_dicts, client_data_sizes)
#
#         # # 计算全局模型参数平均值
#         # global_param_mean = {name: param.mean().item() for name, param in global_model.named_parameters()}
#         # print(f"🔄 轮 {r + 1} 结束后，全局模型参数均值: {global_param_mean}")
#
#         # 评估模型
#         loss, accuracy = evaluate(global_model, test_loader)
#         global_accuracies.append(accuracy)
#         print(f"📊 测试集损失: {loss:.4f} | 测试集准确率: {accuracy:.2f}%")
#
#         # 记录数据到 CSV
#         if grc and hasattr(select_clients, 'latest_weights'):
#             w_loss = select_clients.latest_weights[0]
#             w_diff = select_clients.latest_weights[1]
#             print(f"📈 Round {r + 1} | GRC 权重: w_loss = {w_loss:.4f}, w_diff = {w_diff:.4f}")
#
#         else:
#             w_loss = 'NA'
#             w_diff = 'NA'
#
#         csv_data.append([
#             r + 1,
#             accuracy,
#             total_comm,
#             ",".join(map(str, selected_clients)),
#             w_loss,
#             w_diff
#         ])
#         df = pd.DataFrame(csv_data, columns=[
#             'Round', 'Accuracy', 'Total communication counts', 'Selected Clients',
#             'GRC Weight - Loss', 'GRC Weight - Diff'])
#         df.to_csv(csv_filename, index=False)
#
#     # 输出最终模型的性能
#     final_loss, final_accuracy = evaluate(global_model, test_loader)
#     print(f"\n🎯 Loss of final model test dataset: {final_loss:.4f}")
#     print(f"🎯 Final model test set accuracy: {final_accuracy:.2f}%")
#
#     # 输出通信记录
#     print("\n Client Communication Statistics:")
#     for client_id, counts in communication_counts.items():
#         print(
#             f"Client {client_id}: Sent {counts['send']} times, Received {counts['receive']} times, Completed full_round {counts['full_round']} times")
#
#     T2 = time.time()
#     print('程序运行时间:%s秒' % ((T2 - T1)))
#
#     # 可视化全局模型准确率 vs 轮次
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, rounds + 1), global_accuracies, marker='o', linestyle='-', color='b', label="Test Accuracy")
#     plt.xlabel("Federated Learning Rounds")
#     plt.ylabel("Accuracy")
#     plt.title("Test Accuracy Over Federated Learning Rounds")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#     # 可视化全局模型准确率 vs 客户端完整通信次数
#     plt.figure(figsize=(8, 5))
#     plt.plot(total_communication_counts, global_accuracies, marker='s', linestyle='-', color='r',
#              label="Test Accuracy vs. Communication")
#     plt.xlabel("Total Communication Count per Round")
#     plt.ylabel("Accuracy")
#     plt.title("Test Accuracy vs. Total Communication")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
#
#
#
# if __name__ == "__main__":
#
#     main()
#     #1927.4226


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import numpy as np
# import random
# import os
# import matplotlib.pyplot as plt
# import csv
# import pandas as pd
# from datetime import datetime
# import time
#
#
# # 定义 MLP 模型
# class MLPModel(nn.Module):
#     def __init__(self):
#         super(MLPModel, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 1024)  # 第一层，输入维度 784 -> 200
#         self.fc2 = nn.Linear(1024, 1024)  # 第二层，200 -> 200
#         self.fc3 = nn.Linear(1024, 10)  # 输出层，200 -> 10
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # 展平输入 (batch_size, 1, 28, 28) -> (batch_size, 784)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # 直接输出，不使用 Softmax（因为 PyTorch 的 CrossEntropyLoss 里已经包含了）
#         return x
#
#
#
# class LoRALayer(nn.Module):
#     def __init__(self, in_features, out_features, rank=4, alpha=1):
#         super(LoRALayer, self).__init__()
#         self.rank = rank
#         self.alpha = alpha
#         self.scaling = alpha / rank
#         self.A = nn.Parameter(torch.zeros(in_features, rank))
#         self.B = nn.Parameter(torch.zeros(rank, out_features))
#
#         nn.init.normal_(self.A, mean=0.0, std=0.02)
#         nn.init.zeros_(self.B)
#
#     def forward(self, x):
#         return x @ (self.A @ self.B) * self.scaling
#
#
#
# class LoRAMLPModel(nn.Module):
#     def __init__(self, base_model, rank=4, alpha=1):
#         super(LoRAMLPModel, self).__init__()
#         self.base_model = base_model
#         for param in base_model.parameters():
#             param.requires_grad = False
#
#         self.lora_fc1 = LoRALayer(28 * 28, 1024, rank=rank, alpha=alpha)
#         self.lora_fc2 = LoRALayer(1024, 1024, rank=rank, alpha=alpha)
#         self.lora_fc3 = LoRALayer(1024, 10, rank=rank, alpha=alpha)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # 展平输入
#
#         # 前向传播结合基础模型和 LoRA 部分
#         fc1_out = self.base_model.relu(self.base_model.fc1(x) + self.lora_fc1(x))
#         fc2_out = self.base_model.relu(self.base_model.fc2(fc1_out) + self.lora_fc2(fc1_out))
#         out = self.base_model.fc3(fc2_out) + self.lora_fc3(fc2_out)
#
#         return out
#
#
# # 加载 MNIST 数据集
# def load_mnist_data(data_path="./data"):
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#
#     if os.path.exists(os.path.join(data_path, "MNIST/raw/train-images-idx3-ubyte")):
#         print("✅ MNIST 数据集已存在，跳过下载。")
#     else:
#         print("⬇️ 正在下载 MNIST 数据集...")
#
#     train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
#     test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
#
#     # visualize_mnist_samples(train_data)
#     return train_data, test_data
#
#
# # 显示数据集示例图片
# def visualize_mnist_samples(dataset, num_samples=10):
#     fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.2, 1.5))
#     for i in range(num_samples):
#         img, label = dataset[i]
#         axes[i].imshow(img.squeeze(), cmap="gray")
#         axes[i].set_title(label)
#         axes[i].axis("off")
#     plt.show()
#
#
# def split_data_by_label(dataset, num_clients=10):
#     client_data_sizes = {
#         0: {0: 600},
#         1: {1: 700},
#         2: {2: 500},
#         3: {3: 600},
#         4: {4: 600},
#         5: {5: 500},
#         6: {6: 100},
#         7: {7: 100},
#         8: {8: 100},
#         9: {9: 100}
#     }
#
#     # 统计每个类别的数据索引
#     label_to_indices = {i: [] for i in range(10)}  # 记录每个类别的索引
#     for idx, (_, label) in enumerate(dataset):
#         label_to_indices[label].append(idx)
#
#     # 初始化客户端数据存储
#     client_data_subsets = {}
#     client_actual_sizes = {i: {label: 0 for label in range(10)} for i in range(num_clients)}  # 记录实际分配的数据量
#
#     # 遍历每个客户端，为其分配指定类别的数据
#     for client_id, label_info in client_data_sizes.items():
#         selected_indices = []  # 临时存储该客户端所有选中的索引
#         for label, size in label_info.items():
#             # 确保不超出类别数据集实际大小
#             available_size = len(label_to_indices[label])
#             sample_size = min(available_size, size)
#
#             if sample_size < size:
#                 print(f"⚠️ 警告：类别 {label} 的数据不足，客户端 {client_id} 只能获取 {sample_size} 条样本（需求 {size} 条）")
#
#             # 从该类别中随机抽取样本
#             sampled_indices = random.sample(label_to_indices[label], sample_size)
#             selected_indices.extend(sampled_indices)
#
#             # 记录实际分配的数据量
#             client_actual_sizes[client_id][label] = sample_size
#
#         # 创建 PyTorch Subset
#         client_data_subsets[client_id] = torch.utils.data.Subset(dataset, selected_indices)
#
#     # 打印每个客户端的实际分配数据量
#     print("\n📊 每个客户端实际数据分布:")
#     for client_id, label_sizes in client_actual_sizes.items():
#         print(f"客户端 {client_id}: {label_sizes}")
#
#     return client_data_subsets, client_actual_sizes
#
#
# # 本地训练函数 (用于正常的联邦训练，训练所有参数)
# def local_train(model, train_loader, epochs=5, lr=0.01):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     model.train()
#     for epoch in range(epochs):
#         for batch_x, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#     return model.state_dict()
#
#
# # LoRA训练函数 (只训练LoRA参数，用于客户端选择阶段)
# def local_train_lora(base_model, train_loader, epochs=2, lr=0.01, rank=4, alpha=1):
#     # 创建LoRA模型
#     lora_model = LoRAMLPModel(base_model, rank=rank, alpha=alpha)
#
#     criterion = nn.CrossEntropyLoss()
#     # 只优化LoRA参数
#     optimizer = optim.SGD([p for n, p in lora_model.named_parameters() if p.requires_grad], lr=lr)
#
#     lora_model.train()
#     loss_sq_sum = 0.0
#
#     for epoch in range(epochs):
#         for batch_x, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = lora_model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             loss_sq_sum += loss.item() ** 2
#
#     h_i = loss_sq_sum  # 累积loss平方和
#     return lora_model, h_i
#
#
# # 实时记录训练过程中的loss用做FedGRA (使用LoRA)
# def local_train_fedgra_loss_lora(model, train_loader, epochs=2, lr=0.01, rank=4, alpha=1):
#     # 使用LoRA模型进行轻量级训练
#     lora_model, h_i = local_train_lora(model, train_loader, epochs=epochs, lr=lr, rank=rank, alpha=alpha)
#     return lora_model, h_i
#
#
# #  联邦平均聚合函数
# def fed_avg(global_model, client_state_dicts, client_sizes):
#     global_dict = global_model.state_dict()
#     subkey = [sublist[0] for sublist in client_state_dicts]
#     new_client_sizes = dict(([(key, client_sizes[key]) for key in subkey]))
#     total_data = sum(sum(label_sizes.values()) for label_sizes in new_client_sizes.values())  # 计算所有客户端数据总量
#     for key in global_dict.keys():
#         global_dict[key] = sum(
#             client_state[key] * (sum(new_client_sizes[client_id].values()) / total_data)
#             for (client_id, client_state) in client_state_dicts
#         )
#     global_model.load_state_dict(global_dict)
#     return global_model
#
#
# # 评估模型
# def evaluate(model, test_loader):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     correct, total, total_loss = 0, 0, 0.0
#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += batch_y.size(0)
#             correct += (predicted == batch_y).sum().item()
#     accuracy = correct / total * 100
#     return total_loss / len(test_loader), accuracy
#
#
# # 熵权法实现
# def entropy_weight(l):
#     l = np.array(l)
#
#     # Step 1: Min-Max 归一化（避免负值和爆炸）
#     X_norm = (l - l.min(axis=1, keepdims=True)) / (l.max(axis=1, keepdims=True) - l.min(axis=1, keepdims=True) + 1e-12)
#
#     # Step 2: 转为概率矩阵 P_ki
#     P = X_norm / (X_norm.sum(axis=1, keepdims=True) + 1e-12)
#
#     # Step 3: 计算熵
#     K = 1 / np.log(X_norm.shape[1])
#     E = -K * np.sum(P * np.log(P + 1e-12), axis=1)  # shape: (2,)
#
#     # Step 4: 计算信息效用值 & 权重
#     d = 1 - E
#     weights = d / np.sum(d)
#     return weights.tolist()
#
#
# # 灰色关联度实现 (使用LoRA模型)
# def calculate_GRC(global_model, client_lora_models, client_losses):
#     """
#     计算 GRC 分数 + 熵权法权重
#     使用LoRA模型中的差异计算GRC
#     """
#     # 计算参数差异（只考虑LoRA部分的参数）
#     param_diffs = []
#     for lora_model in client_lora_models:
#         # 提取所有LoRA参数形成一个向量
#         lora_params = []
#         for name, param in lora_model.named_parameters():
#             if param.requires_grad:  # 只考虑LoRA部分参数
#                 lora_params.append(param.detach().view(-1))
#
#         if lora_params:
#             lora_vec = torch.cat(lora_params)
#             # 用LoRA参数的L2范数作为差异度量
#             diff = torch.norm(lora_vec).item()
#             param_diffs.append(diff)
#         else:
#             param_diffs.append(0.0)
#
#     # 2. 映射原始指标到 [0, 1] 区间（为熵权法准备）
#     def map_sequence_loss(sequence):
#         max_val, min_val = max(sequence), min(sequence)
#         denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
#         return [(max_val - x) / denom for x in sequence]  # 越小越好，负相关
#
#     def map_sequence_diff(sequence):
#         max_val, min_val = max(sequence), min(sequence)
#         denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
#         return [(x - min_val) / denom for x in sequence]  # 越大越好，正相关
#
#     # 用于 GRC 的映射
#     mapped_losses = map_sequence_loss(client_losses)
#     mapped_diffs = map_sequence_diff(param_diffs)
#
#     # 3. 熵权法计算权重（根据映射值，非 GRC）
#     grc_metrics = np.vstack([mapped_losses, mapped_diffs])  # shape: (2, n_clients)
#     weights = entropy_weight(grc_metrics)  # w_loss, w_diff
#
#     # 4. 计算 GRC 分数（ξki），参考值为 1
#     ref_loss, ref_diff = 1.0, 1.0
#     delta_losses = [abs(x - ref_loss) for x in mapped_losses]
#     delta_diffs = [abs(x - ref_diff) for x in mapped_diffs]
#     all_deltas = delta_losses + delta_diffs
#     max_delta, min_delta = max(all_deltas), min(all_deltas)
#
#     grc_losses = []
#     grc_diffs = []
#     rho = 0.5  # 区分度因子
#     for d_loss, d_diff in zip(delta_losses, delta_diffs):
#         grc_loss = (min_delta + rho * max_delta) / (d_loss + rho * max_delta)
#         grc_diff = (min_delta + rho * max_delta) / (d_diff + rho * max_delta)
#         grc_losses.append(grc_loss)
#         grc_diffs.append(grc_diff)
#
#     # 5. 加权求和得到最终 GRC 分数
#     grc_losses = np.array(grc_losses)
#     grc_diffs = np.array(grc_diffs)
#     weighted_score = grc_losses * weights[0] + grc_diffs * weights[1]
#
#     # 调试（每个客户端的 loss、diff、得分）
#     print("\n GRC得分]")
#     for i in range(len(client_lora_models)):
#         print(f"Client {i} | loss: {client_losses[i]:.4f}, diff: {param_diffs[i]:.4f}, "
#               f"mapped_loss: {mapped_losses[i]:.4f}, mapped_diff: {mapped_diffs[i]:.4f}, "
#               f"GRC: {weighted_score[i]:.4f}")
#     print(f"熵权法权重: w_loss = {weights[0]:.4f}, w_diff = {weights[1]:.4f}")
#
#     return weighted_score, weights
#
#
# # 客户端选择器 (使用LoRA)
# def select_clients(client_loaders, use_all_clients=False, num_select=None,
#                    select_by_loss=False, global_model=None, grc=False,
#                    fairness_tracker=None, lora_rank=4, lora_alpha=1):
#     if grc:  # 使用 GRC 选择客户端
#         client_lora_models = []
#         # 1. 使用LoRA训练本地模型并计算损失 (轻量级训练)
#         client_losses = []
#         for client_id, client_loader in client_loaders.items():
#             # 使用LoRA训练 - 减少训练成本
#             trained_lora_model, h_i = local_train_fedgra_loss_lora(
#                 global_model, client_loader, epochs=2, lr=0.001, rank=lora_rank, alpha=lora_alpha
#             )
#             client_lora_models.append(trained_lora_model)
#             client_losses.append(h_i)
#
#         # 2. 计算 GRC 分数
#         grc_scores, grc_weights = calculate_GRC(global_model, client_lora_models, client_losses)
#         select_clients.latest_weights = grc_weights  # 记录权重
#
#         # 3. 按 GRC 分数排序（从高到低，GRC越高表示越好）
#         client_grc_pairs = list(zip(client_loaders.keys(), grc_scores))
#         client_grc_pairs.sort(key=lambda x: x[1], reverse=True)  # 降序排序
#
#         # 4. 选择 GRC 最高的前 num_select 个客户端
#         selected_clients = [client_id for client_id, _ in client_grc_pairs[:num_select]]
#         return selected_clients
#
#     # 其余选择逻辑保持不变
#
#     if use_all_clients is True:
#         print("Selecting all clients")
#         return list(client_loaders.keys())
#
#     if num_select is None:
#         raise ValueError("If use_all_clients=False, num_select cannot be None!")
#
#     if select_by_loss and global_model:
#         client_losses = {}
#
#         for client_id, loader in client_loaders.items():
#             local_model = MLPModel()
#             local_model.load_state_dict(global_model.state_dict())
#             local_train(local_model, loader, epochs=5, lr=0.01)
#             loss, _ = evaluate(local_model, loader)
#             client_losses[client_id] = loss
#         selected_clients = sorted(client_losses, key=client_losses.get, reverse=True)[:num_select]
#         print(f"Selected {num_select} clients with the highest loss: {selected_clients}")
#     else:
#         selected_clients = random.sample(list(client_loaders.keys()), num_select)
#         print(f"Randomly selected {num_select} clients: {selected_clients}")
#
#     return selected_clients
#
#
# def update_communication_counts(communication_counts, selected_clients, event):
#     for client_id in selected_clients:
#         communication_counts[client_id][event] += 1
#
#         if event == "send" and communication_counts[client_id]['receive'] > 0:
#             communication_counts[client_id]['full_round'] += 1
#
#
# def main():
#     torch.manual_seed(0)
#     random.seed(0)
#     np.random.seed(0)
#
#     # 加载 MNIST 数据集
#     train_data, test_data = load_mnist_data()
#
#     # 生成客户端数据集，每个客户端包含多个类别
#     client_datasets, client_data_sizes = split_data_by_label(train_data)
#
#     # 创建数据加载器
#     client_loaders = {client_id: data.DataLoader(dataset, batch_size=32, shuffle=True)
#                       for client_id, dataset in client_datasets.items()}
#     test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)
#
#     # 初始化全局模型
#     global_model = MLPModel()
#     global_accuracies = []  # 记录每轮全局模型的测试集准确率
#     total_communication_counts = []  # 记录每轮客户端通信次数
#     rounds = 300  # 联邦学习轮数
#     use_all_clients = False  # 是否进行客户端选择
#     num_selected_clients = 2  # 每轮选择客户端训练数量
#     use_loss_based_selection = False  # 是否根据 loss 选择客户端
#     grc = True
#
#     # LoRA超参数
#     lora_rank = 4  # LoRA秩
#     lora_alpha = 16  # LoRA缩放因子
#
#     # 初始化通信计数器
#     communication_counts = {}
#     for client_id in client_loaders.keys():
#         communication_counts[client_id] = {
#             'send': 0,
#             'receive': 0,
#             'full_round': 0
#         }
#
#     # 实验数据存储 CSV
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv_filename = f"training_data_lora_{timestamp}.csv"
#     csv_data = []
#
#     for r in range(rounds):
#         print(f"\n🔄 第 {r + 1} 轮聚合")
#         # 选择客户端 (使用LoRA减少计算成本)
#         selected_clients = select_clients(
#             client_loaders,
#             use_all_clients=use_all_clients,
#             num_select=num_selected_clients,
#             select_by_loss=use_loss_based_selection,
#             global_model=global_model,
#             grc=grc,
#             lora_rank=lora_rank,
#             lora_alpha=lora_alpha
#         )
#
#         # 记录客户端接收通信次数
#         update_communication_counts(communication_counts, selected_clients, "receive")
#         client_state_dicts = []
#
#         # 客户端本地训练 (正常训练所有参数)
#         for client_id in selected_clients:
#             client_loader = client_loaders[client_id]
#             local_model = MLPModel()
#             local_model.load_state_dict(global_model.state_dict())  # 复制全局模型参数
#             local_state = local_train(local_model, client_loader, epochs=1, lr=0.01)  # 训练5轮
#             client_state_dicts.append((client_id, local_state))  # 存储 (客户端ID, 训练后的参数)
#
#             update_communication_counts(communication_counts, [client_id], "send")  # 记录客户端上报通信次数
#
#             param_mean = {name: param.mean().item() for name, param in local_model.named_parameters()}
#             print(f"  ✅ 客户端 {client_id} 训练完成 | 样本数量: {sum(client_data_sizes[client_id].values())}")
#             print(f"  📌 客户端 {client_id} 模型参数均值: {param_mean}")
#
#         # 计算本轮通信次数
#         total_send = sum(
#             communication_counts[c]['send'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
#         total_receive = sum(
#             communication_counts[c]['receive'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
#         total_comm = total_send + total_receive  # 每轮独立的总通信次数
#
#         # 如果不是第一轮，累加前一轮的通信次数
#         if len(total_communication_counts) > 0:
#             total_comm += total_communication_counts[-1]
#         total_communication_counts.append(total_comm)
#
#         # 聚合模型参数
#         global_model = fed_avg(global_model, client_state_dicts, client_data_sizes)
#
#         # 评估模型
#         loss, accuracy = evaluate(global_model, test_loader)
#         global_accuracies.append(accuracy)
#         print(f"📊 测试集损失: {loss:.4f} | 测试集准确率: {accuracy:.2f}%")
#
#         # 记录数据到 CSV
#         if grc and hasattr(select_clients, 'latest_weights'):
#             w_loss = select_clients.latest_weights[0]
#             w_diff = select_clients.latest_weights[1]
#             print(f"📈 Round {r + 1} | GRC 权重: w_loss = {w_loss:.4f}, w_diff = {w_diff:.4f}")
#
#         else:
#             w_loss = 'NA'
#             w_diff = 'NA'
#
#         csv_data.append([
#             r + 1,
#             accuracy,
#             total_comm,
#             ",".join(map(str, selected_clients)),
#             w_loss,
#             w_diff
#         ])
#         df = pd.DataFrame(csv_data, columns=[
#             'Round', 'Accuracy', 'Total communication counts', 'Selected Clients',
#             'GRC Weight - Loss', 'GRC Weight - Diff'])
#         df.to_csv(csv_filename, index=False)
#
#     # 输出最终模型的性能
#     final_loss, final_accuracy = evaluate(global_model, test_loader)
#     print(f"\n🎯 Loss of final model test dataset: {final_loss:.4f}")
#     print(f"🎯 Final model test set accuracy: {final_accuracy:.2f}%")
#
#     # 输出通信记录
#     print("\n Client Communication Statistics:")
#     for client_id, counts in communication_counts.items():
#         print(
#             f"Client {client_id}: Sent {counts['send']} times, Received {counts['receive']} times, Completed full_round {counts['full_round']} times")
#
#
#
# if __name__ == "__main__":
#     T1 = time.time()
#     main()
#     T2 = time.time()
#     print('程序运行时间:%s秒' % ((T2 - T1)))
#     #1320.884345293045
#
#     # 1553.40735
#
#
#
# # -*- coding: utf-8 -*-
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# loss_csv_300 = "training_data_lora_20250507_104136.csv"
# grc_csv_300 = "training_data_20250507_115745.csv"
#
#
# df_loss_300 = pd.read_csv(loss_csv_300)
# df_grc_300 = pd.read_csv(grc_csv_300)
#
#
# plt.figure(figsize=(8, 5))
# plt.plot(df_loss_300['Round'], df_loss_300['Accuracy'], label='GRC-based-with-lora', color='blue')
# plt.plot(df_grc_300['Round'], df_grc_300['Accuracy'], label='GRC-based', color='red')
# plt.xlabel("Round")
# plt.ylabel("Test Accuracy (%)")
# plt.title("Accuracy over Rounds (300 rounds)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# -*- coding: utf-8 -*-
"""Federated Training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tE-M1T-9BL-HglL5A7asx4b31Sdyhcdq
"""

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import numpy as np
# import random
# import os
# import matplotlib.pyplot as plt
# import ssl
# from datetime import datetime
# import pandas as pd
# import torch.nn.functional as F
# import time
#
#
# class LoRALayer(nn.Module):
#     def __init__(self, in_dim, out_dim, rank, alpha, use_svd=False, linear=None):
#         super().__init__()
#         self.use_svd = use_svd  # Flag to decide if SVD should be used
#         self.alpha = alpha
#
#         if use_svd:
#             # Initialize LoRA parameters using SVD
#             source_linear = linear.weight.data
#             source_linear = source_linear.float()
#             U, S, V = torch.linalg.svd(source_linear)  # Perform SVD
#             U_r = U[:, :rank]  # Take the first 'rank' singular vectors
#             S_r = torch.diag(S[:rank])  # Create the diagonal matrix for the top 'rank' singular values
#             V_r = V[:, :rank].t()  # Take the first 'rank' singular vectors
#
#             # Use SVD components to initialize A and B
#             self.A = nn.Parameter(U_r.contiguous())  # The A matrix
#             self.B = nn.Parameter((S_r @ V_r).contiguous())  # The B matrix (rank x out_dim) using S_r * V_r^T
#         else:
#             # Random initialization
#             std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
#             self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
#             self.B = nn.Parameter(torch.zeros(rank, out_dim))
#
#     # def forward(self, x):
#     #     x=self.alpha*(x@self.A@self.B)
#     #     return x
#
#
# class LinearWithLoRA(nn.Module):
#     def __init__(self, linear, rank, alpha, use_svd=False):
#         super().__init__()
#         self.linear = linear
#         self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, use_svd=use_svd, linear=linear)
#
#     def forward(self, x):
#         # Apply LoRA to original weights
#         lora = self.lora.A @ self.lora.B  # combine LoRA metrices
#         combined_weight = self.linear.weight + self.lora.alpha * lora
#         return F.linear(x, combined_weight, self.linear.bias)
#
#
# # 定义 MLP 模型
# class MLPModel(nn.Module):
#     def __init__(self, is_LoRA=False, rank=4, alpha=8, use_svd=False):
#         super(MLPModel, self).__init__()
#
#         self.is_LoRA = is_LoRA
#         self.use_svd = use_svd
#
#         if not self.is_LoRA:  # Use the original linear layers
#             self.fc1 = nn.Linear(28 * 28, 1024)  # 第一层，输入维度 784 -> 200
#             self.fc2 = nn.Linear(1024, 1024)  # 第二层，200 -> 200
#             self.fc3 = nn.Linear(1024, 10)  # 输出层，200 -> 10
#             self.relu = nn.ReLU()
#         else:  # Use LoRA-enhanced layers
#             self.fc1 = nn.Linear(28 * 28, 1024)
#             self.fc2 = LinearWithLoRA(nn.Linear(1024, 1024), rank, alpha, use_svd=self.use_svd)
#             self.fc3 = nn.Linear(1024, 10)
#             self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # Direct output, no need for Softmax (CrossEntropyLoss already includes it)
#         return x
#
#
# def freeze_linear_layers(model):
#     """Freeze the original layers in the model, leaving only LoRA layers trainable."""
#     for child in model.children():
#         if isinstance(child, LinearWithLoRA):
#             # Freeze the original weights (linear layers)
#             for param in child.linear.parameters():
#                 param.requires_grad = False  # not accepting additional gradients
#         else:
#             # Recursively freeze linear layers in children modules
#             freeze_linear_layers(child)
#
#
# # 加载 MNIST 数据集
# def load_mnist_data(data_path="./data"):
#     # Temporarily Skip SSL velidation step
#     ssl._create_default_https_context = ssl._create_unverified_context
#
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#
#     if os.path.exists(os.path.join(data_path, "MNIST/raw/train-images-idx3-ubyte")):
#         print("✅ MNIST 数据集已存在，跳过下载。")
#     else:
#         print("⬇️ 正在下载 MNIST 数据集...")
#
#     train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
#     test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
#
#     return train_data, test_data
#
#
# # 分割 MNIST 数据，使每个客户端只包含某个数字类别
# def split_data_by_label(dataset, num_clients=10):
#     # Mannually set each client'id and corresponding dataset distribution
#     # client_data_sizes = {
#     #     0: {0: 600, 1: 700, 2: 600, 3: 600, 4: 500, 5: 500, 6: 100, 7: 100, 8: 100, 9: 100},
#     #     1: {0: 700, 1: 600, 2: 600, 3: 600, 4: 500, 5: 100, 6: 100, 7: 100, 8: 100, 9: 600},
#     #     2: {0: 500, 1: 600, 2: 700, 3: 600, 4: 100, 5: 100, 6: 100, 7: 100, 8: 600, 9: 500},
#     #     3: {0: 600, 1: 600, 2: 500, 3: 100, 4: 100, 5: 100, 6: 100, 7: 500, 8: 500, 9: 700},
#     #     4: {0: 600, 1: 500, 2: 100, 3: 100, 4: 100, 5: 100, 6: 600, 7: 700, 8: 500, 9: 500},
#     #     5: {0: 500, 1: 100, 2: 100, 3: 100, 4: 100, 5: 600, 6: 500, 7: 600, 8: 700, 9: 600},
#     #     6: {0: 100, 1: 100, 2: 100, 3: 100, 4: 700, 5: 500, 6: 600, 7: 500, 8: 500, 9: 600},
#     #     7: {0: 100, 1: 100, 2: 100, 3: 600, 4: 500, 5: 600, 6: 500, 7: 600, 8: 500, 9: 100},
#     #     8: {0: 100, 1: 100, 2: 500, 3: 500, 4: 600, 5: 500, 6: 600, 7: 500, 8: 100, 9: 100},
#     #     9: {0: 100, 1: 700, 2: 600, 3: 600, 4: 600, 5: 500, 6: 600, 7: 100, 8: 100, 9: 100}
#     # }
#
#     client_data_sizes = {
#         0: {0: 600},
#         1: {1: 600},
#         2: {2: 500},
#         3: {3: 600},
#         4: {4: 600},
#         5: {5: 500},
#         6: {6: 100},
#         7: {7: 100},
#         8: {8: 100},
#         9: {9: 100}
#     }
#
#     # Initialize an empty dictionary to store indices for each label (from 0 to 9)
#     label_to_indices = {}
#
#     for label in range(10):
#         label_to_indices[label] = []
#
#         # Loop through the dataset using enumerate to get both the index and the data item.
#     # Each data item is a tuple (image, label).
#     for index, (_, label) in enumerate(dataset):
#         # Append the current index to the list corresponding to the data's label.
#         label_to_indices[label].append(index)
#
#     # Create an empty dictionary to store the data subset for each client.
#     client_data_subsets = {}
#
#     # Initialize a dictionary to record the actual number of samples allocated for each label in each client.
#     client_actual_sizes = {}
#     for client_id in range(num_clients):
#         # For each client, initialize an empty dictionary to store the sample counts for labels 0 to 9.
#         client_actual_sizes[client_id] = {}
#
#         # For each label from 0 to 9, set the initial count to 0.
#         for label in range(10):
#             client_actual_sizes[client_id][label] = 0
#
#     # Iterate over each client and assign data for the specified labels.
#     for client_id, label_info in client_data_sizes.items():
#         selected_indices = []
#
#         for label, required_size in label_info.items():
#             available_size = len(label_to_indices[label])
#
#             # Determine the number of samples to select
#             sample_size = min(available_size, required_size)
#
#             # If the available sample size is less than the required size, print a warning message.
#             if sample_size < required_size:
#                 print(
#                     f"⚠️ Warning: Not enough data for label {label}. Client {client_id} can only get {sample_size} samples (required {required_size}).")
#
#             # Randomly select the determined number of indices and add the selected indices to the client's list.
#             sampled_indices = random.sample(label_to_indices[label], sample_size)
#             selected_indices.extend(sampled_indices)
#
#             # Record the actual number of samples allocated for this label for the current client.
#             client_actual_sizes[client_id][label] = sample_size
#
#         # Create a PyTorch Subset for this client using the selected indices.
#         client_data_subsets[client_id] = torch.utils.data.Subset(dataset, selected_indices)
#
#     print("\n📊 Actual data distribution per client:")
#     for client_id, label_sizes in client_actual_sizes.items():
#         print(f"Client {client_id}: {label_sizes}")
#
#     # Return both the client data subsets and the dictionary of actual sample sizes.
#     return client_data_subsets, client_actual_sizes
#
#
# def local_train(model, train_loader, epochs=5, lr=0.1, is_LoRA=False, freeze_W=False):
#     """Train the model on a local client, freezing original layers if using LoRA."""
#     criterion = nn.CrossEntropyLoss()
#
#     if is_LoRA and freeze_W:  # If LoRA and freeze_W is enabled, freeze the original layers
#         freeze_linear_layers(model)
#         # Only update LoRA parameters (A and B)
#         params_to_update = [param for param in model.parameters() if param.requires_grad]
#         optimizer = optim.SGD(params_to_update, lr=lr)
#     elif is_LoRA and not freeze_W:
#         # Update LoRA parameters (A and B) and Weight
#         params_to_update = [param for param in model.parameters() if param.requires_grad]
#         optimizer = optim.SGD(params_to_update, lr=lr)
#     else:
#         optimizer = optim.SGD(model.parameters(), lr=lr)
#
#     model.train()
#     for epoch in range(epochs):
#         for batch_x, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#
#     return model.state_dict()
#
#
# # 联邦平均聚合函数
# def fed_avg(global_model, client_state_dicts, client_sizes):
#     global_dict = global_model.state_dict()
#
#     # Extract client IDs from client_state_dicts tuples.
#     subkey = [sublist[0] for sublist in client_state_dicts]
#
#     # Create a new dictionary of client sizes using only the clients that were selected.
#     new_client_sizes = dict([(key, client_sizes[key]) for key in subkey])
#
#     # Calculate the total number of samples across all selected clients.
#     # Here, each client size is now assumed to be a nested dictionary (per label), so we sum the values for each client.
#     total_data = sum(sum(label_sizes.values()) for label_sizes in new_client_sizes.values())
#
#     # Update each parameter in the global model.
#     for key in global_dict.keys():
#         global_dict[key] = sum(
#             client_state[key] * (sum(new_client_sizes[client_id].values()) / total_data)
#             for (client_id, client_state) in client_state_dicts
#         )
#
#     global_model.load_state_dict(global_dict)
#     return global_model
#
#
# # 评估模型
# def evaluate(model, test_loader):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     correct, total, total_loss = 0, 0, 0.0
#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += batch_y.size(0)
#             correct += (predicted == batch_y).sum().item()
#     accuracy = correct / total * 100
#     return total_loss / len(test_loader), accuracy
#
#
# # 熵权法实现
#
# def entropy_weight(l):
#     l = np.array(l)
#
#     # Step 1: Min-Max 归一化（避免负值和爆炸）
#     X_norm = (l - l.min(axis=1, keepdims=True)) / (l.max(axis=1, keepdims=True) - l.min(axis=1, keepdims=True) + 1e-12)
#
#     # Step 2: 转为概率矩阵 P_ki
#     P = X_norm / (X_norm.sum(axis=1, keepdims=True) + 1e-12)
#
#     # Step 3: 计算熵
#     K = 1 / np.log(X_norm.shape[1])
#     E = -K * np.sum(P * np.log(P + 1e-12), axis=1)  # shape: (2,)
#
#     # Step 4: 计算信息效用值 & 权重
#     d = 1 - E
#     weights = d / np.sum(d)
#     return weights.tolist()
#
#
# # 灰色关联度实现
# def calculate_GRC(global_model, client_models, client_losses):
#     """
#     计算 GRC 分数 + 熵权法权重
#     修正：
#       - 映射顺序错误
#       - 熵权法使用错误指标
#     """
#     # 正确写法：使用整体参数向量计算 L2 范数（符合文献）
#     param_diffs = []
#     for model in client_models:
#         global_vec = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
#
#         local_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
#         diff = torch.norm(local_vec - global_vec).item()
#         param_diffs.append(diff)
#
#     # 2. 映射原始指标到 [0, 1] 区间（为熵权法准备）
#     def map_sequence_loss(sequence):
#         max_val, min_val = max(sequence), min(sequence)
#         denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
#         return [(max_val - x) / denom for x in sequence]  # 越小越好，负相关
#
#     def map_sequence_diff(sequence):
#         max_val, min_val = max(sequence), min(sequence)
#         denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
#         return [(x - min_val) / denom for x in sequence]  # 越大越好，正相关
#
#     # 用于 GRC 的映射
#     mapped_losses = map_sequence_loss(client_losses)
#     mapped_diffs = map_sequence_diff(param_diffs)
#
#     # 3. 熵权法计算权重（根据映射值，非 GRC）
#     grc_metrics = np.vstack([mapped_losses, mapped_diffs])  # shape: (2, n_clients)
#     weights = entropy_weight(grc_metrics)  # w_loss, w_diff
#
#     # 4. 计算 GRC 分数（ξki），参考值为 1
#     ref_loss, ref_diff = 1.0, 1.0
#     delta_losses = [abs(x - ref_loss) for x in mapped_losses]
#     delta_diffs = [abs(x - ref_diff) for x in mapped_diffs]
#     all_deltas = delta_losses + delta_diffs
#     max_delta, min_delta = max(all_deltas), min(all_deltas)
#
#     grc_losses = []
#     grc_diffs = []
#     rho = 0.5  # 区分度因子
#     for d_loss, d_diff in zip(delta_losses, delta_diffs):
#         grc_loss = (min_delta + rho * max_delta) / (d_loss + rho * max_delta)
#         grc_diff = (min_delta + rho * max_delta) / (d_diff + rho * max_delta)
#         grc_losses.append(grc_loss)
#         grc_diffs.append(grc_diff)
#
#     # 5. 加权求和得到最终 GRC 分数
#     grc_losses = np.array(grc_losses)
#     grc_diffs = np.array(grc_diffs)
#     weighted_score = grc_losses * weights[0] + grc_diffs * weights[1]
#
#     # 调试（每个客户端的 loss、diff、得分）
#     print("\n GRC得分]")
#     for i in range(len(client_models)):
#         print(f"Client {i} | loss: {client_losses[i]:.4f}, diff: {param_diffs[i]:.4f}, "
#               f"mapped_loss: {mapped_losses[i]:.4f}, mapped_diff: {mapped_diffs[i]:.4f}, "
#               f"GRC: {weighted_score[i]:.4f}")
#     print(f"熵权法权重: w_loss = {weights[0]:.4f}, w_diff = {weights[1]:.4f}")
#
#     return weighted_score, weights
#
#
# def select_clients(client_loaders, use_all_clients=False, num_select=None,
#                    select_by_loss=False, global_model=None, grc=True, is_LoRA=False, use_svd=False, freeze_W=False):
#     if grc:  # 使用 GRC 选择客户端
#         client_models = []
#         # 1. 训练本地模型并计算损失
#         client_losses = []
#         for client_id, client_loader in client_loaders.items():
#             local_model = MLPModel(is_LoRA=is_LoRA, use_svd=use_svd)  # Define if using LoRA or SVD
#             local_model.load_state_dict(global_model.state_dict())  # 同步全局模型
#             local_train(local_model, client_loader, epochs=1, lr=0.01, is_LoRA=is_LoRA,
#                         freeze_W=freeze_W)  # Define if using LoRA
#             client_models.append(local_model)
#             loss, _ = evaluate(local_model, client_loader)
#             client_losses.append(loss)
#
#         # 2. 计算 GRC 分数
#         grc_scores, grc_weights = calculate_GRC(global_model, client_models, client_losses)
#         select_clients.latest_weights = grc_weights  # 记录权重
#
#         # 3. 按 GRC 分数排序（从高到低，GRC越高表示越好）
#         client_grc_pairs = list(zip(client_loaders.keys(), grc_scores))
#         client_grc_pairs.sort(key=lambda x: x[1], reverse=True)  # 降序排序
#
#         # 4. 选择 GRC 最高的前 num_select 个客户端
#         selected = [client_id for client_id, _ in client_grc_pairs[:num_select]]
#         return selected
#
#     # 其余选择逻辑保持不变
#     if use_all_clients is True:
#         print("Selecting all clients")
#         return list(client_loaders.keys())
#
#     if num_select is None:
#         raise ValueError("If use_all_clients=False, num_select cannot be None!")
#
#     if select_by_loss and global_model:
#         client_losses = {}
#         for client_id, loader in client_loaders.items():
#             loss, _ = evaluate(global_model, loader)
#             client_losses[client_id] = loss
#
#         selected_clients = sorted(client_losses, key=client_losses.get, reverse=True)[:num_select]
#         print(f"Selected {num_select} clients with the highest loss: {selected_clients}")
#     else:
#         selected_clients = random.sample(list(client_loaders.keys()), num_select)
#         print(f"Randomly selected {num_select} clients: {selected_clients}")
#
#     return selected_clients
#
#
# def update_communication_counts(communication_counts, selected_clients, event):
#     """
#     客户端通信计数
#     - event='receive' 表示客户端接收到全局模型
#     - event='send' 表示客户端上传本地模型
#     - event='full_round' 仅在客户端完成完整收发时增加
#     """
#     for client_id in selected_clients:
#         communication_counts[client_id][event] += 1
#
#         # 仅当客户端完成一次完整的 send 和 receive 时增加 full_round
#         if event == "send" and communication_counts[client_id]['receive'] > 0:
#             communication_counts[client_id]['full_round'] += 1
#
#
# def run_experiment(selection_method, rounds=100, num_selected_clients=2, is_LoRA=False, use_svd=False, freeze_W=False):
#     torch.manual_seed(0)
#     random.seed(0)
#     np.random.seed(0)
#
#     # 加载 MNIST 数据集
#     train_data, test_data = load_mnist_data()
#
#     # 生成客户端数据集，每个客户端只包含特定类别
#     client_datasets, client_data_sizes = split_data_by_label(train_data)
#
#     # 创建数据加载器
#     client_loaders = {client_id: data.DataLoader(dataset, batch_size=32, shuffle=True)
#                       for client_id, dataset in client_datasets.items()}
#     test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)
#
#     # Initialize global model, communication_counts, and results storage
#     global_model = MLPModel(is_LoRA=is_LoRA, use_svd=use_svd)  # Use LoRA & SVD or not
#     global_accuracies = []
#     total_communication_counts = []
#     csv_data = []
#
#     # Initialize communication counters for all clients
#     communication_counts = {client_id: {'send': 0, 'receive': 0, 'full_round': 0}
#                             for client_id in client_loaders.keys()}
#
#     for r in range(rounds):
#         print(f"\nRound {r + 1}")
#
#         # Select clients based on the specified method:
#         if selection_method == "fedGRA":
#             selected_clients = select_clients(client_loaders, use_all_clients=False,
#                                               num_select=num_selected_clients,
#                                               select_by_loss=True, global_model=global_model, grc=True, is_LoRA=is_LoRA,
#                                               use_svd=use_svd)
#         elif selection_method == "high_loss":
#             # Use loss-based selection without GRC (select top 2 highest loss clients)
#             selected_clients = select_clients(client_loaders, use_all_clients=False,
#                                               num_select=num_selected_clients,
#                                               select_by_loss=True, global_model=global_model, grc=False,
#                                               is_LoRA=is_LoRA, use_svd=use_svd, freeze_W=freeze_W)
#         elif selection_method == "fedavg":
#             # Use FedAvg with either random selection or all clients.
#             # Using all clients or random selection for FedAvg.
#             selected_clients = select_clients(client_loaders, use_all_clients=True,
#                                               num_select=num_selected_clients,
#                                               select_by_loss=False, global_model=global_model, grc=False)
#
#         # Record receive communication count
#         update_communication_counts(communication_counts, selected_clients, "receive")
#         client_state_dicts = []
#
#         # Perform local training on selected clients
#         for client_id in selected_clients:
#             client_loader = client_loaders[client_id]
#             local_model = MLPModel(is_LoRA=is_LoRA, use_svd=use_svd)  # Use LoRA & SVD or not
#             local_model.load_state_dict(global_model.state_dict())  # Sync with the global model
#             local_train(local_model, client_loader, epochs=1, lr=0.01, is_LoRA=is_LoRA)
#             client_state_dicts.append((client_id, local_model.state_dict()))
#             update_communication_counts(communication_counts, [client_id], "send")
#             print(f"Client {client_id} trained.")
#
#         # Compute communication counts for this round
#         total_send = sum(communication_counts[c]['send'] - (communication_counts[c]['full_round'] - 1)
#                          for c in selected_clients)
#         total_receive = sum(communication_counts[c]['receive'] - (communication_counts[c]['full_round'] - 1)
#                             for c in selected_clients)
#         total_comm = total_send + total_receive
#         total_communication_counts.append(total_comm)
#
#         # Aggregate model updates
#         global_model = fed_avg(global_model, client_state_dicts, client_data_sizes)
#
#         # Evaluate global model
#         loss, accuracy = evaluate(global_model, test_loader)
#         global_accuracies.append(accuracy)
#         print(f"Test Accuracy: {accuracy:.2f}%")
#
#         # Save round data; add a column indicating the method used if desired.
#         csv_data.append([r + 1, accuracy, total_comm])
#
#     # Convert collected data to a DataFrame
#     df = pd.DataFrame(csv_data, columns=['Round', f'Accuracy_{selection_method}', f'Comm_{selection_method}'])
#     return df
#
#
# def main_experiments():
#     rounds = 300
#     # Run experiments for each method
#     # df_high_loss = run_experiment("high_loss", rounds, num_selected_clients=2)
#     # df_lora = run_experiment("fedGRA", rounds, num_selected_clients=2, is_LoRA=True, freeze_W=True)
#     # df_lora_svd = run_experiment("high_loss", rounds, num_selected_clients=2, is_LoRA=True, use_svd=True, freeze_W=True)
#     df_lora_svd = run_experiment("fedGRA", rounds, num_selected_clients=2, is_LoRA=True, use_svd=True, freeze_W=True)
#
#     # Merge DataFrames on 'Round'
#     # df_combined = df_lora.merge(df_lora_svd, on='Round').merge(df_lora_W, on='Round')
#     df_combined = df_lora_svd
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv_filename = f"training_data_lora_{timestamp}.csv"
#     # Save to CSV for later inspection if needed
#     df_combined.to_csv(csv_filename, index=False)
#
#
#
# if __name__ == "__main__":
#     T1 = time.time()
#     main_experiments()
#     T2 = time.time()
#     print('程序运行时间:%s秒' % ((T2 - T1)))
#     # 737 ("fedGRA", rounds, num_selected_clients=2, is_LoRA=True, use_svd=True, freeze_W=True)
#     # 1312.913296699524
#     # 程序运行时间: 2809.2965700626373秒

# 浮点数计算包 参与计算的总参数量 线性训练时间 svd每个epoch只做一次

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
import time


# 定义 MLP 模型
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)  # 第一层，输入维度 784 -> 200
        self.fc2 = nn.Linear(200, 200)  # 第二层，200 -> 200
        self.fc3 = nn.Linear(200, 10)  # 输出层，200 -> 10
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入 (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 直接输出，不使用 Softmax（因为 PyTorch 的 CrossEntropyLoss 里已经包含了）
        return x


# SVD缓存类 - 新增优化
class SVDCache:
    def __init__(self):
        self.cache = {}  # 格式: {layer_name: (U_r, S_r, V_r)}

    def get_svd(self, layer_name, linear_layer, rank):
        """获取SVD结果，如果缓存中存在则直接返回，否则计算并缓存"""
        if layer_name in self.cache:
            return self.cache[layer_name]

        # 计算SVD
        W = linear_layer.weight.data.float()
        U, S, V = torch.svd(W)

        # 截断
        U_r = U[:, :rank]  # [in_features, rank]
        S_r = torch.diag(S[:rank])  # [rank, rank]
        V_r = V.T[:rank, :]  # [rank, out_features]

        # 缓存结果
        self.cache[layer_name] = (U_r, S_r, V_r)
        return U_r, S_r, V_r


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1, use_svd=False, base_linear=None, svd_cache=None,
                 layer_name=None):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_svd = use_svd
        self.relu = nn.ReLU()

        # Store the base layer's bias if it exists
        self.base_bias = None
        if base_linear is not None and base_linear.bias is not None:
            self.base_bias = base_linear.bias.detach().clone()

        if self.use_svd and base_linear is not None and svd_cache is not None:
            # 使用缓存的SVD结果初始化LoRA参数
            U_r, S_r, V_r = svd_cache.get_svd(layer_name, base_linear, rank)

            # 使用SVD结果初始化A和B
            A = U_r  # [out_features, rank]
            B = (S_r @ V_r)  # [rank, in_features]

            self.A = nn.Parameter(A.contiguous())
            self.B = nn.Parameter(B.contiguous())

        else:
            self.A = nn.Parameter(torch.zeros(in_features, rank))
            self.B = nn.Parameter(torch.zeros(rank, out_features))
            nn.init.normal_(self.A, mean=0.0, std=0.02)
            nn.init.zeros_(self.B)

    def forward(self, x):
        AB = self.A @ self.B  # [in, rank] @ [rank, out]
        output =AB @ x.T
        if self.base_bias is not None:
            output += self.base_bias.unsqueeze(1)
        return output.T


# 定义带 LoRA 的 MLP 模型
class LoRAMLPModel(nn.Module):
    def __init__(self, base_model, rank=4, alpha=1, use_svd=True, svd_cache=None):
        super(LoRAMLPModel, self).__init__()
        self.base_model = base_model

        for param in base_model.parameters():
            param.requires_grad = False

        # 使用SVD缓存
        self.lora_fc1 = LoRALayer(28 * 28, 200, rank=rank, alpha=alpha, use_svd=use_svd,
                                  base_linear=base_model.fc1, svd_cache=svd_cache, layer_name='fc1')
        self.lora_fc2 = LoRALayer(200, 200, rank=rank, alpha=alpha, use_svd=use_svd,
                                  base_linear=base_model.fc2, svd_cache=svd_cache, layer_name='fc2')
        # self.lora_fc3 = LoRALayer(28 * 28, 10, rank=rank, alpha=alpha, use_svd=use_svd,
        #                           base_linear=base_model.fc3, svd_cache=svd_cache, layer_name='fc3')
        self.fc3_2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入

        # 前向传播结合基础模型和 LoRA 部分
        fc1_out = self.lora_fc1.relu(self.lora_fc1(x))
        fc2_out = self.lora_fc2.relu(self.lora_fc2(fc1_out))
        out = self.fc3_2(fc2_out)

        return out


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


# 显示数据集示例图片
def visualize_mnist_samples(dataset, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.2, 1.5))
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(label)
        axes[i].axis("off")
    plt.show()


def split_data_by_label(dataset, num_clients=10):
    client_data_sizes = {
        0: {0: 600},
        1: {1: 600},
        2: {2: 600},
        3: {3: 600},
        4: {4: 600},
        5: {5: 600},
        6: {6: 600},
        7: {7: 600},
        8: {8: 600},
        9: {9: 600}
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


# 本地训练函数 (用于正常的联邦训练，训练所有参数)
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


# LoRA训练函数 (只训练LoRA参数，用于客户端选择阶段)
def local_train_lora(base_model, train_loader, epochs=2, lr=0.01, rank=4, alpha=1, svd_cache=None):
    # 创建LoRA模型
    lora_model = LoRAMLPModel(base_model, rank=rank, alpha=alpha, svd_cache=svd_cache)

    criterion = nn.CrossEntropyLoss()
    # 只优化LoRA参数
    optimizer = optim.SGD([p for n, p in lora_model.named_parameters() if p.requires_grad], lr=lr)

    lora_model.train()
    loss_sq_sum = 0.0

    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = lora_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            loss_sq_sum += loss.item() ** 2

    h_i = loss_sq_sum  # 累积loss平方和
    return lora_model, h_i


# 实时记录训练过程中的loss用做FedGRA (使用LoRA)
def local_train_fedgra_loss_lora(model, train_loader, epochs=2, lr=0.01, rank=4, alpha=1, svd_cache=None):
    # 使用LoRA模型进行轻量级训练
    lora_model, h_i = local_train_lora(model, train_loader, epochs=epochs, lr=lr, rank=rank, alpha=alpha,
                                       svd_cache=svd_cache)
    return lora_model, h_i


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


# 熵权法实现
def entropy_weight(l):
    l = np.array(l)

    # Step 1: Min-Max 归一化（避免负值和爆炸）
    X_norm = (l - l.min(axis=1, keepdims=True)) / (l.max(axis=1, keepdims=True) - l.min(axis=1, keepdims=True) + 1e-12)

    # Step 2: 转为概率矩阵 P_ki
    P = X_norm / (X_norm.sum(axis=1, keepdims=True) + 1e-12)

    # Step 3: 计算熵
    K = 1 / np.log(X_norm.shape[1])
    E = -K * np.sum(P * np.log(P + 1e-12), axis=1)  # shape: (2,)

    # Step 4: 计算信息效用值 & 权重
    d = 1 - E
    weights = d / np.sum(d)
    return weights.tolist()


# 灰色关联度实现 (使用LoRA模型)
def calculate_GRC(global_model, client_lora_models, client_losses, initial_lora_params=None):
    """
    计算 GRC 分数 + 熵权法权重
    使用LoRA模型中的训练前后差异计算GRC - 修改为先计算AB乘积再计算差异

    Args:
        global_model: 全局模型
        client_lora_models: 训练后的LoRA模型列表
        client_losses: 客户端损失列表
        initial_lora_params: 训练前的LoRA参数字典 {name: param_tensor}
    """
    # 计算参数差异（先计算AB乘积，再计算差异）
    param_diffs = []

    for trained_lora_model in client_lora_models:
        # 如果提供了初始参数，则计算与初始参数的差异
        if initial_lora_params is not None:
            # 收集差异向量
            diff_vectors = []

            # 获取模型中所有的LoRA层
            lora_layers = {
                'lora_fc1': trained_lora_model.lora_fc1,
                'lora_fc2': trained_lora_model.lora_fc2,
                # 'lora_fc3': trained_lora_model.lora_fc3
            }

            for layer_name, lora_layer in lora_layers.items():
                # 计算当前训练后的AB乘积
                current_AB = lora_layer.A @ lora_layer.B  # [in_features, out_features]

                # 构建训练前的AB乘积
                A_name = f"{layer_name}.A"
                B_name = f"{layer_name}.B"

                if A_name in initial_lora_params and B_name in initial_lora_params:
                    initial_A = initial_lora_params[A_name]
                    initial_B = initial_lora_params[B_name]
                    initial_AB = initial_A @ initial_B

                    # 计算AB乘积的差异
                    diff = current_AB - initial_AB
                    diff_vectors.append(diff.view(-1))

            if diff_vectors:
                # 连接所有差异向量并计算L2范数
                diff_vec = torch.cat(diff_vectors)
                diff = torch.norm(diff_vec, 2).item()
                param_diffs.append(diff)
            else:
                param_diffs.append(0.0)
        else:
            # 如果没有提供初始参数，计算当前LoRA层AB乘积的L2范数
            diff_vectors = []

            # 获取模型中所有的LoRA层
            lora_layers = {
                'lora_fc1': trained_lora_model.lora_fc1,
                'lora_fc2': trained_lora_model.lora_fc2,
                'lora_fc3': trained_lora_model.lora_fc3
            }

            for layer_name, lora_layer in lora_layers.items():
                # 计算当前训练后的AB乘积
                current_AB = lora_layer.A @ lora_layer.B  # [in_features, out_features]
                diff_vectors.append(current_AB.view(-1))

            if diff_vectors:
                # 连接所有AB乘积并计算L2范数
                diff_vec = torch.cat(diff_vectors)
                diff = torch.norm(diff_vec, 2).item()
                param_diffs.append(diff)
            else:
                param_diffs.append(0.0)

    # 2. 映射原始指标到 [0, 1] 区间（为熵权法准备）
    def map_sequence_loss(sequence):
        max_val, min_val = max(sequence), min(sequence)
        denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
        return [(max_val - x) / denom for x in sequence]  # 越小越好，负相关

    def map_sequence_diff(sequence):
        max_val, min_val = max(sequence), min(sequence)
        denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
        return [(x - min_val) / denom for x in sequence]  # 越大越好，正相关

    # 用于 GRC 的映射
    mapped_losses = map_sequence_loss(client_losses)
    mapped_diffs = map_sequence_diff(param_diffs)

    # 3. 熵权法计算权重（根据映射值，非 GRC）
    grc_metrics = np.vstack([mapped_losses, mapped_diffs])  # shape: (2, n_clients)
    weights = entropy_weight(grc_metrics)  # w_loss, w_diff

    # 4. 计算 GRC 分数（ξki），参考值为 1
    ref_loss, ref_diff = 1.0, 1.0
    delta_losses = [abs(x - ref_loss) for x in mapped_losses]
    delta_diffs = [abs(x - ref_diff) for x in mapped_diffs]
    all_deltas = delta_losses + delta_diffs
    max_delta, min_delta = max(all_deltas), min(all_deltas)

    grc_losses = []
    grc_diffs = []
    rho = 0.5  # 区分度因子
    for d_loss, d_diff in zip(delta_losses, delta_diffs):
        grc_loss = (min_delta + rho * max_delta) / (d_loss + rho * max_delta)
        grc_diff = (min_delta + rho * max_delta) / (d_diff + rho * max_delta)
        grc_losses.append(grc_loss)
        grc_diffs.append(grc_diff)

    # 5. 加权求和得到最终 GRC 分数
    grc_losses = np.array(grc_losses)
    grc_diffs = np.array(grc_diffs)
    weighted_score = grc_losses * weights[0] + grc_diffs * weights[1]

    # 调试（每个客户端的 loss、diff、得分）
    print("\n GRC得分]")
    for i in range(len(client_lora_models)):
        print(f"Client {i} | loss: {client_losses[i]:.4f}, diff: {param_diffs[i]:.4f}, "
              f"mapped_loss: {mapped_losses[i]:.4f}, mapped_diff: {mapped_diffs[i]:.4f}, "
              f"GRC: {weighted_score[i]:.4f}")
    print(f"熵权法权重: w_loss = {weights[0]:.4f}, w_diff = {weights[1]:.4f}")

    return weighted_score, weights


# 客户端选择器 (使用LoRA)
def select_clients(client_loaders, use_all_clients=False, num_select=None,
                   select_by_loss=False, global_model=None, grc=False,
                   fairness_tracker=None, lora_rank=4, lora_alpha=1):
    if grc:  # 使用 GRC 选择客户端
        client_lora_models = []

        # 创建SVD缓存 - 优化点: 只计算并缓存一次SVD结果
        svd_cache = SVDCache()

        # 1. 只创建一个初始LoRA模型作为基准
        initial_lora_model = LoRAMLPModel(global_model, rank=lora_rank, alpha=lora_alpha, svd_cache=svd_cache)

        # 存储初始LoRA参数状态
        initial_lora_params = {}
        for name, param in initial_lora_model.named_parameters():
            if param.requires_grad:  # 只保存LoRA部分的参数
                initial_lora_params[name] = param.clone().detach()

        # 2. 使用LoRA训练本地模型并计算损失 (轻量级训练)
        client_losses = []
        for client_id, client_loader in client_loaders.items():
            # 使用LoRA训练 - 减少训练成本，并传递SVD缓存
            trained_lora_model, h_i = local_train_fedgra_loss_lora(
                global_model, client_loader, epochs=5, lr=0.0005,
                rank=lora_rank, alpha=lora_alpha, svd_cache=svd_cache
            )
            client_lora_models.append(trained_lora_model)
            client_losses.append(h_i)

        # 3. 计算 GRC 分数，现在传递初始LoRA参数状态
        grc_scores, grc_weights = calculate_GRC(global_model, client_lora_models, client_losses, initial_lora_params)
        select_clients.latest_weights = grc_weights  # 记录权重

        # 4. 按 GRC 分数排序（从高到低，GRC越高表示越好）
        client_grc_pairs = list(zip(client_loaders.keys(), grc_scores))
        client_grc_pairs.sort(key=lambda x: x[1], reverse=True)  # 降序排序

        # 5. 选择 GRC 最高的前 num_select 个客户端
        selected_clients = [client_id for client_id, _ in client_grc_pairs[:num_select]]
        return selected_clients

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
            local_train(local_model, loader, epochs=5, lr=0.01)
            loss, _ = evaluate(local_model, loader)
            client_losses[client_id] = loss
        selected_clients = sorted(client_losses, key=client_losses.get, reverse=True)[:num_select]
        print(f"Selected {num_select} clients with the highest loss: {selected_clients}")
    else:
        selected_clients = random.sample(list(client_loaders.keys()), num_select)
        print(f"Randomly selected {num_select} clients: {selected_clients}")

    return selected_clients


def update_communication_counts(communication_counts, selected_clients, event):
    for client_id in selected_clients:
        communication_counts[client_id][event] += 1

        if event == "send" and communication_counts[client_id]['receive'] > 0:
            communication_counts[client_id]['full_round'] += 1


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # 加载 MNIST 数据集
    train_data, test_data = load_mnist_data()

    # 生成客户端数据集，每个客户端包含多个类别
    client_datasets, client_data_sizes = split_data_by_label(train_data)

    # 创建数据加载器
    client_loaders = {client_id: data.DataLoader(dataset, batch_size=32, shuffle=True)
                      for client_id, dataset in client_datasets.items()}
    test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)

    # 初始化全局模型
    global_model = MLPModel()
    global_accuracies = []  # 记录每轮全局模型的测试集准确率
    total_communication_counts = []  # 记录每轮客户端通信次数
    rounds = 100  # 联邦学习轮数
    use_all_clients = False  # 是否进行客户端选择
    num_selected_clients = 2  # 每轮选择客户端训练数量
    use_loss_based_selection = False  # 是否根据 loss 选择客户端
    grc = True

    # LoRA超参数
    lora_rank = 180  # LoRA秩
    lora_alpha = 16  # LoRA缩放因子

    # 初始化通信计数器
    communication_counts = {}
    for client_id in client_loaders.keys():
        communication_counts[client_id] = {
            'send': 0,
            'receive': 0,
            'full_round': 0
        }

    # 实验数据存储 CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_data_lora_{timestamp}.csv"
    csv_data = []

    for r in range(rounds):
        print(f"\n🔄 第 {r + 1} 轮聚合")
        # 选择客户端 (使用LoRA减少计算成本)
        selected_clients = select_clients(
            client_loaders,
            use_all_clients=use_all_clients,
            num_select=num_selected_clients,
            select_by_loss=use_loss_based_selection,
            global_model=global_model,
            grc=grc,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )

        # 记录客户端接收通信次数
        update_communication_counts(communication_counts, selected_clients, "receive")
        client_state_dicts = []

        # 客户端本地训练 (正常训练所有参数)
        for client_id in selected_clients:
            client_loader = client_loaders[client_id]
            local_model = MLPModel()
            local_model.load_state_dict(global_model.state_dict())  # 复制全局模型参数
            local_state = local_train(local_model, client_loader, epochs=2, lr=0.01)  # 训练5轮
            client_state_dicts.append((client_id, local_state))  # 存储 (客户端ID, 训练后的参数)

            update_communication_counts(communication_counts, [client_id], "send")  # 记录客户端上报通信次数

            param_mean = {name: param.mean().item() for name, param in local_model.named_parameters()}
            print(f"  ✅ 客户端 {client_id} 训练完成 | 样本数量: {sum(client_data_sizes[client_id].values())}")
            print(f"  📌 客户端 {client_id} 模型参数均值: {param_mean}")

        # 计算本轮通信次数
        total_send = sum(
            communication_counts[c]['send'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
        total_receive = sum(
            communication_counts[c]['receive'] - (communication_counts[c]['full_round'] - 1) for c in selected_clients)
        total_comm = total_send + total_receive  # 每轮独立的总通信次数

        # 如果不是第一轮，累加前一轮的通信次数
        if len(total_communication_counts) > 0:
            total_comm += total_communication_counts[-1]
        total_communication_counts.append(total_comm)

        # 聚合模型参数
        global_model = fed_avg(global_model, client_state_dicts, client_data_sizes)

        # 评估模型
        loss, accuracy = evaluate(global_model, test_loader)
        global_accuracies.append(accuracy)
        print(f"📊 测试集损失: {loss:.4f} | 测试集准确率: {accuracy:.2f}%")

        # 记录数据到 CSV
        if grc and hasattr(select_clients, 'latest_weights'):
            w_loss = select_clients.latest_weights[0]
            w_diff = select_clients.latest_weights[1]
            print(f"📈 Round {r + 1} | GRC 权重: w_loss = {w_loss:.4f}, w_diff = {w_diff:.4f}")

        else:
            w_loss = 'NA'
            w_diff = 'NA'

        csv_data.append([
            r + 1,
            accuracy,
            total_comm,
            ",".join(map(str, selected_clients)),
            w_loss,
            w_diff
        ])
        df = pd.DataFrame(csv_data, columns=[
            'Round', 'Accuracy', 'Total communication counts', 'Selected Clients',
            'GRC Weight - Loss', 'GRC Weight - Diff'])
        df.to_csv(csv_filename, index=False)

    # 输出最终模型的性能
    final_loss, final_accuracy = evaluate(global_model, test_loader)
    print(f"\n🎯 Loss of final model test dataset: {final_loss:.4f}")
    print(f"🎯 Final model test set accuracy: {final_accuracy:.2f}%")

    # 输出通信记录
    print("\n Client Communication Statistics:")
    for client_id, counts in communication_counts.items():
        print(
            f"Client {client_id}: Sent {counts['send']} times, Received {counts['receive']} times, Completed full_round {counts['full_round']} times")

    # # 可视化全局模型准确率 vs 轮次
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, rounds + 1), global_accuracies, marker='o', linestyle='-', color='b', label="Test Accuracy")
    # plt.xlabel("Federated Learning Rounds")
    # plt.ylabel("Accuracy")
    # plt.title("Test Accuracy Over Federated Learning Rounds")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # 可视化全局模型准确率 vs 客户端完整通信次数
    # plt.figure(figsize=(8, 5))
    # plt.plot(total_communication_counts, global_accuracies, marker='s', linestyle='-', color='r',
    #          label="Test Accuracy vs. Communication")
    # plt.xlabel("Total Communication Count per Round")
    # plt.ylabel("Accuracy")
    # plt.title("Test Accuracy vs. Total Communication")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    T1 = time.time()
    main()
    T2 = time.time()
    print('程序运行时间:%s秒' % ((T2 - T1)))
    # 1520.884345293045
    # 100 453.0890119075775秒  程序运行时间:440.32077145576477秒


