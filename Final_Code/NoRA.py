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
from fvcore.nn import FlopCountAnalysis # library for calculating flops


# Define MLP Model
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)  # first layer 784 -> 200
        self.fc2 = nn.Linear(200, 200)  # second layer，200 -> 200
        self.fc3 = nn.Linear(200, 10)  # output layer，200 -> 10
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flaten inputs (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) 
        return x


# SVD cache class
class SVDCache:
    def __init__(self):
        self.cache = {}  # format: {layer_name: (U_r, S_r, V_r)}

    def get_svd(self, layer_name, linear_layer, rank):
        """Obtain the SVD result. If it exists in the cache, return it directly; otherwise, calculate and cache it"""
        if layer_name in self.cache:
            return self.cache[layer_name]

        # compute SVD
        W = linear_layer.weight.data.float()
        U, S, V = torch.svd(W)

        U_r = U[:, :rank]  # [in_features, rank]
        S_r = torch.diag(S[:rank])  # [rank, rank]
        V_r = V.T[:rank, :]  # [rank, out_features]

        # Cache results 
        self.cache[layer_name] = (U_r, S_r, V_r)
        return U_r, S_r, V_r


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1, use_svd=False, base_linear=None, svd_cache=None,
                 layer_name=None, r_in = 4):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_svd = use_svd
        self.relu = nn.ReLU()
        self.r_in = r_in

        self.base_bias = None
        if base_linear is not None and base_linear.bias is not None:
            self.base_bias = base_linear.bias.detach().clone()

        if self.use_svd and base_linear is not None and svd_cache is not None:
            U_r, S_r, V_r = svd_cache.get_svd(layer_name, base_linear, rank)

            # sqrt_S = torch.sqrt(S_r)  # [rank, rank] 对角矩阵
            # A = U_r @ sqrt_S  # [out_features, rank]
            # B = sqrt_S @ V_r  # [rank, in_features]

            A_outer = U_r.clone()  # [in_features, r_out]
            B_outer = V_r.clone()  # [r_out, out_features]

            S_diag = torch.sqrt(S_r)  # [r_out, r_out]

            # Step 5: initializing inner LoRA
            if rank % r_in == 0:
                step_size = rank // r_in
                A_inner = S_diag[:, ::step_size][:, :r_in]  # [r_out, r_in]
                B_inner = S_diag[::step_size, :][:r_in, :]  # [r_in, r_out]
            else:
                A_inner = S_diag[:, :r_in]  # [r_out, r_in]
                B_inner = S_diag[:r_in, :]  # [r_in, r_out]

            self.A_inner = nn.Parameter(A_inner.contiguous())
            self.B_inner = nn.Parameter(B_inner.contiguous())
            self.A_outer = nn.Parameter(A_outer.contiguous(), requires_grad=False)
            self.B_outer = nn.Parameter(B_outer.contiguous(), requires_grad=False)

        else:
            self.A = nn.Parameter(torch.zeros(in_features, rank))
            self.B = nn.Parameter(torch.zeros(rank, out_features))
            nn.init.normal_(self.A, mean=0.0, std=0.02)
            nn.init.zeros_(self.B)

    def forward(self, x):
        AB = self.A_outer @ self.A_inner @ self.B_inner @ self.B_outer
        output =AB @ x.T
        if self.base_bias is not None:
            output += self.base_bias.unsqueeze(1)
        return output.T


# define LoRA based MLP Model
class LoRAMLPModel(nn.Module):
    def __init__(self, base_model, rank=4, alpha=1, use_svd=True, svd_cache=None,r_in = 4):
        super(LoRAMLPModel, self).__init__()
        self.base_model = base_model

        for param in base_model.parameters():
            param.requires_grad = False

        # Utilise SVD cache
        self.lora_fc1 = LoRALayer(28 * 28, 200, rank=rank, alpha=alpha, use_svd=use_svd,
                                  base_linear=base_model.fc1, svd_cache=svd_cache, layer_name='fc1',r_in = r_in)
        self.lora_fc2 = LoRALayer(200, 200, rank=rank, alpha=alpha, use_svd=use_svd,
                                  base_linear=base_model.fc2, svd_cache=svd_cache, layer_name='fc2',r_in = r_in)
        # self.lora_fc3 = LoRALayer(28 * 28, 10, rank=rank, alpha=alpha, use_svd=use_svd,
        #                           base_linear=base_model.fc3, svd_cache=svd_cache, layer_name='fc3',r_in = r_in)
        self.fc3_2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input

        # Forward propagation combines the basic model and the LoRA part
        fc1_out = self.lora_fc1.relu(self.lora_fc1(x))
        fc2_out = self.lora_fc2.relu(self.lora_fc2(fc1_out))
        out = self.fc3_2(fc2_out)

        return out


# loading MNIST data
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


# Display sample images of the dataset
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

    # Count the data index of each category
    label_to_indices = {i: [] for i in range(10)}  # Record the index of each category
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    # Initialize the client data store 
    client_data_subsets = {}
    client_actual_sizes = {i: {label: 0 for label in range(10)} for i in range(num_clients)}  # Record the actual amount of data allocated

    # Traverse each client and assign data of the specified category to it
    for client_id, label_info in client_data_sizes.items():
        selected_indices = []  # Temporarily store all the selected indexes of this client
        for label, size in label_info.items():
            # Make sure not to exceed the actual size of the category dataset
            available_size = len(label_to_indices[label])
            sample_size = min(available_size, size)

            if sample_size < size:
                print(f"⚠️ 警告：类别 {label} 的数据不足，客户端 {client_id} 只能获取 {sample_size} 条样本（需求 {size} 条）")

            # Randomly draw samples from this category
            sampled_indices = random.sample(label_to_indices[label], sample_size)
            selected_indices.extend(sampled_indices)

            # Record the actual amount of data allocated
            client_actual_sizes[client_id][label] = sample_size

        # Create PyTorch Subset
        client_data_subsets[client_id] = torch.utils.data.Subset(dataset, selected_indices)

    # Print the actual amount of allocated data for each client
    print("\n📊 每个客户端实际数据分布:")
    for client_id, label_sizes in client_actual_sizes.items():
        print(f"客户端 {client_id}: {label_sizes}")

    return client_data_subsets, client_actual_sizes


# Local training function (for normal federated training, training all parameters)
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


# LoRA training function (Only training LoRA parameters, used in the client selection stage)
def local_train_lora(base_model, train_loader, epochs=2, lr=0.01, rank=4, alpha=1, svd_cache=None,r_in = 4):
    lora_model = LoRAMLPModel(base_model, rank=rank, alpha=alpha, svd_cache=svd_cache,r_in = r_in)

    criterion = nn.CrossEntropyLoss()
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

    h_i = loss_sq_sum  # accumulate loss square sum 
    return lora_model, h_i


# Real-time recording of the loss during the training process for use as FedGRA (using LoRA)
def local_train_fedgra_loss_lora(model, train_loader, epochs=2, lr=0.01, rank=4, alpha=1, svd_cache=None,r_in = 4):
    lora_model, h_i = local_train_lora(model, train_loader, epochs=epochs, lr=lr, rank=rank, alpha=alpha,
                                       svd_cache=svd_cache,r_in = r_in)
    return lora_model, h_i


#  Federal average aggregation function
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


# Evaluate Model 
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


# Implemented the entropy weight method
def entropy_weight(l):
    l = np.array(l)

    # Step 1: Min-Max normalization (to avoid negative values and explosions)
    X_norm = (l - l.min(axis=1, keepdims=True)) / (l.max(axis=1, keepdims=True) - l.min(axis=1, keepdims=True) + 1e-12)

    # Step 2: Convert to a probability matrix P_ki
    P = X_norm / (X_norm.sum(axis=1, keepdims=True) + 1e-12)

    # Step 3: Calculate Entropy
    K = 1 / np.log(X_norm.shape[1])
    E = -K * np.sum(P * np.log(P + 1e-12), axis=1)  # shape: (2,)

    # Step 4: Calculate the information utility value & weight
    d = 1 - E
    weights = d / np.sum(d)
    return weights.tolist()


def calculate_GRC(global_model, client_lora_models, client_losses, initial_lora_params=None):
    """
    Compute GRC scores + entropy-based weights.
    Use the difference between pre- and post-training LoRA parameters to compute GRC scores.
    Modified to first compute the AB product, then calculate the difference.

    Args:
        global_model: The global model used for aggregation.
        client_lora_models: List of LoRA models after client-side training.
        client_losses: List of training losses for each client.
        initial_lora_params: Dictionary of initial LoRA parameters before training {name: param_tensor}.
    """

    param_diffs = []

    for trained_lora_model in client_lora_models:
        if initial_lora_params is not None:
            diff_vectors = []
            
            # Collect LoRA layers
            lora_layers = {
                'lora_fc1': trained_lora_model.lora_fc1,
                'lora_fc2': trained_lora_model.lora_fc2,
                # 'lora_fc3': trained_lora_model.lora_fc3
            }

            for layer_name, lora_layer in lora_layers.items():
                current_AB = lora_layer.A @ lora_layer.B  # [in_features, out_features]
                A_name = f"{layer_name}.A"
                B_name = f"{layer_name}.B"

                if A_name in initial_lora_params and B_name in initial_lora_params:
                    initial_A = initial_lora_params[A_name]
                    initial_B = initial_lora_params[B_name]
                    initial_AB = initial_A @ initial_B

                    # Compute difference between current and initial AB products
                    diff = current_AB - initial_AB
                    diff_vectors.append(diff.view(-1))

            if diff_vectors:
                # Concatenate all difference vectors and compute L2 norm
                diff_vec = torch.cat(diff_vectors)
                diff = torch.norm(diff_vec, 2).item()
                param_diffs.append(diff)
            else:
                param_diffs.append(0.0)
        else:
            # If no initial parameters are provided, compute L2 norm of current AB products
            diff_vectors = []

            # Collect LoRA layers from the model
            lora_layers = {
                'lora_fc1': trained_lora_model.lora_fc1,
                'lora_fc2': trained_lora_model.lora_fc2,
                'lora_fc3': trained_lora_model.lora_fc3
            }

            for layer_name, lora_layer in lora_layers.items():
                # Compute current AB product
                current_AB = lora_layer.A @ lora_layer.B  # [in_features, out_features]
                diff_vectors.append(current_AB.view(-1))

            if diff_vectors:
                # Concatenate all AB products and compute L2 norm
                diff_vec = torch.cat(diff_vectors)
                diff = torch.norm(diff_vec, 2).item()
                param_diffs.append(diff)
            else:
                param_diffs.append(0.0)

    # 2. Normalize original metrics to [0, 1] range (for entropy weight method)
    def map_sequence_loss(sequence):
        max_val, min_val = max(sequence), min(sequence)
        denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
        return [(max_val - x) / denom for x in sequence]  # Lower is better → negative correlation

    def map_sequence_diff(sequence):
        max_val, min_val = max(sequence), min(sequence)
        denom = max_val - min_val if abs(max_val - min_val) > 1e-8 else 1e-8
        return [(x - min_val) / denom for x in sequence]  # Higher is better → positive correlation

    # Mapping for GRC
    mapped_losses = map_sequence_loss(client_losses)
    mapped_diffs = map_sequence_diff(param_diffs)

    # 3. Compute weights using Entropy Weight Method (based on mapped values)
    grc_metrics = np.vstack([mapped_losses, mapped_diffs])  # shape: (2, n_clients)
    weights = entropy_weight(grc_metrics)  # w_loss, w_diff

    # 4. Compute GRC scores (ξki), reference value is 1
    ref_loss, ref_diff = 1.0, 1.0
    delta_losses = [abs(x - ref_loss) for x in mapped_losses]
    delta_diffs = [abs(x - ref_diff) for x in mapped_diffs]
    all_deltas = delta_losses + delta_diffs
    max_delta, min_delta = max(all_deltas), min(all_deltas)

    grc_losses = []
    grc_diffs = []
    rho = 0.5  # Distinguishing coefficient
    for d_loss, d_diff in zip(delta_losses, delta_diffs):
        grc_loss = (min_delta + rho * max_delta) / (d_loss + rho * max_delta)
        grc_diff = (min_delta + rho * max_delta) / (d_diff + rho * max_delta)
        grc_losses.append(grc_loss)
        grc_diffs.append(grc_diff)

    # 5. Compute final GRC score via weighted sum
    grc_losses = np.array(grc_losses)
    grc_diffs = np.array(grc_diffs)
    weighted_score = grc_losses * weights[0] + grc_diffs * weights[1]

    # Debug output for each client: loss, diff, GRC score
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
                   fairness_tracker=None, lora_rank=4, lora_alpha=1,r_in = 4):
    if grc:  
        client_lora_models = []

        # create SVD cache - only store once 
        svd_cache = SVDCache()

        # 1. Only create an initial LoRA model as the benchmark
        initial_lora_model = LoRAMLPModel(global_model, rank=lora_rank, alpha=lora_alpha, svd_cache=svd_cache)

        # Store the initial LoRA parameter status
        initial_lora_params = {}
        for name, param in initial_lora_model.named_parameters():
            if param.requires_grad:  # Only save the parameters of the LoRA part
                initial_lora_params[name] = param.clone().detach()

        # 2. Train the local model with LoRA and calculate the loss (lightweight training) 
        client_losses = []
        param_count_this_round = 0 # storing parameter sum in this round 
        flops_this_round = 0 # storing FLOPs sum in this round 
        for client_id, client_loader in client_loaders.items():
            # Train with LoRA - reduce training costs and pass SVD cache
            trained_lora_model, h_i = local_train_fedgra_loss_lora(
                global_model, client_loader, epochs=5, lr=0.0005,
                rank=lora_rank, alpha=lora_alpha, svd_cache=svd_cache,r_in = r_in
            )
            client_lora_models.append(trained_lora_model)
            client_losses.append(h_i)

            # Calculating parameters used in Model
            param_count_this_round += count_nora_parameters(trained_lora_model) # adding up calculated parameters

            # Calculating forward FLOPs
            batch_size = 1
            dummy_input = torch.randn(batch_size, 1, 28, 28).to(next(trained_lora_model.parameters()).device)
            flops = FlopCountAnalysis(trained_lora_model, (dummy_input,))
            forward_flops = flops.total()

            # Estimate total training FLOPs (forward + backward)
            # Assumption: for each batch, forward + backward ≈ 3 × forward FLOPs; for each client: num_batches × num_epochs × 3 × forward FLOPs
            # the number of batches = len(client_loader)
            num_batches = len(client_loader)
            flops_per_client = forward_flops * 3 * num_batches * 5 # 5 = number of epochs
            flops_this_round += flops_per_client

        # 3. Calculate the GRC score and now pass the initial LoRA parameter state
        grc_scores, grc_weights = calculate_GRC(global_model, client_lora_models, client_losses, initial_lora_params)
        select_clients.latest_weights = grc_weights  

        # 4. Sort by GRC score (from high to low, the higher the GRC, the better)
        client_grc_pairs = list(zip(client_loaders.keys(), grc_scores))
        client_grc_pairs.sort(key=lambda x: x[1], reverse=True) 

        # 5. Select the top num_select clients with the highest GRC
        selected_clients = [client_id for client_id, _ in client_grc_pairs[:num_select]]
        return selected_clients, param_count_this_round, flops_this_round

    # Others remain the same 

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

# calculating parameters used in any model 
def count_nora_parameters(model):
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if hasattr(module, "A_outer") and hasattr(module, "A_inner") and hasattr(module, "B_inner") and hasattr(module, "B_outer"):
                total += module.A_outer.numel()
                total += module.A_inner.numel()
                total += module.B_inner.numel()
                total += module.B_outer.numel()
    return total


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
    rounds = 500  # 联邦学习轮数
    use_all_clients = False  # 是否进行客户端选择
    num_selected_clients = 2  # 每轮选择客户端训练数量
    use_loss_based_selection = False  # 是否根据 loss 选择客户端
    grc = True

    # LoRA超参数
    lora_rank = 8  # LoRA秩
    lora_alpha = 16
    r_in = 4

    start_time = time.time() # Starting time 
    accumulated_params = 0
    accumulated_flops = 0

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
        selected_clients, param_count_this_round, flops_this_round = select_clients(
            client_loaders,
            use_all_clients=use_all_clients,
            num_select=num_selected_clients,
            select_by_loss=use_loss_based_selection,
            global_model=global_model,
            grc=grc,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            r_in = r_in
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

        elapsed_time = time.time() - start_time
        accumulated_params += param_count_this_round # adding up parameters
        accumulated_flops += flops_this_round# adding up flops  

        csv_data.append([
            r + 1,
            accuracy,
            total_comm,
            ",".join(map(str, selected_clients)),
            w_loss,
            w_diff,
            round(elapsed_time, 2), 
            accumulated_params, 
            accumulated_flops
        ])

        df = pd.DataFrame(csv_data, columns=[
            'Round', 'Accuracy', 'Total communication counts', 'Selected Clients',
            'GRC Weight - Loss', 'GRC Weight - Diff', 'Elapsed Time (s)', 'Parameter Usage', 'FLOPs'])
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


