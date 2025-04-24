# -*- coding: utf-8 -*-
"""
Federated Learning with GRA Client Selection

This code uses the MNIST dataset and splits it into 10 clients.
In each federated learning round, only 2 clients are selected based on
the GRA (GRC) client selection mechanism.
"""

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
import pandas as pd
from datetime import datetime
import torch.nn.functional as F

# -------------------------------
# Define a Plain MLP Model
# -------------------------------
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        # Three-layer MLP for MNIST
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------
# Data Loading and Splitting Functions
# -------------------------------
def load_mnist_data(data_path="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if os.path.exists(os.path.join(data_path, "MNIST/raw/train-images-idx3-ubyte")):
        print("✅ MNIST dataset exists. Skipping download.")
    else:
        print("⬇️ Downloading MNIST dataset...")
    train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    return train_data, test_data

def split_data_by_label(dataset, num_clients=10):
    """
    Manually split dataset into client subsets.
    Each client is assigned data for a specific label.
    Returns a dictionary mapping client IDs to data subsets and their sizes.
    """
    # Pre-defined client data sizes for 10 clients (one label per client)
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
    # Map each label to its image indices
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)
    client_data_subsets = {}
    client_actual_sizes = {i: {label: 0 for label in range(10)} for i in range(num_clients)}
    # Split data to each client based on the above sizes
    for client_id, label_info in client_data_sizes.items():
        selected_indices = []
        for label, size in label_info.items():
            available_size = len(label_to_indices[label])
            sample_size = min(available_size, size)
            if sample_size < size:
                print(f"⚠️ Warning: Label {label} has only {sample_size} samples for client {client_id} (requested {size}).")
            sampled_indices = random.sample(label_to_indices[label], sample_size)
            selected_indices.extend(sampled_indices)
            client_actual_sizes[client_id][label] = sample_size
        client_data_subsets[client_id] = data.Subset(dataset, selected_indices)
    print("\n📊 Actual data distribution per client:")
    for client_id, label_sizes in client_actual_sizes.items():
        print(f"Client {client_id}: {label_sizes}")
    return client_data_subsets, client_actual_sizes

# -------------------------------
# Federated Learning Utilities
# -------------------------------
def local_train(model, train_loader, epochs=1, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def fed_avg(global_model, client_state_dicts, client_sizes):
    global_dict = global_model.state_dict()
    # Compute the total number of samples from all clients
    total_data = sum(sum(label_sizes.values()) for label_sizes in client_sizes.values())
    for key in global_dict.keys():
        global_dict[key] = sum(
            client_state[key] * (sum(client_sizes[client_id].values()) / total_data)
            for (client_id, client_state) in client_state_dicts
        )
    global_model.load_state_dict(global_dict)
    return global_model

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

def update_communication_counts(communication_counts, selected_clients, event):
    """
    Update the communication counters.
    event: 'receive' when clients get the global model; 'send' when they send local updates.
    """
    for client_id in selected_clients:
        communication_counts[client_id][event] += 1
        # Increase full round when a client has both received and sent
        if event == "send" and communication_counts[client_id]['receive'] > 0:
            communication_counts[client_id]['full_round'] += 1

# -------------------------------
# GRA (GRC) Client Selection Functions
# -------------------------------
def entropy_weight(matrix):
    """Compute entropy weights for the given matrix (each row is a metric)."""
    entropies = []
    for row in matrix:
        row_sum = np.sum(row) + 1e-12  # avoid division by zero
        P = row / row_sum
        K = 1.0 / np.log(len(row))
        E = -K * np.sum(P * np.log(P + 1e-12))
        entropies.append(E)
    information_gain = [1 - e for e in entropies]
    sum_ig = sum(information_gain)
    weights = [ig / sum_ig for ig in information_gain]
    return weights

def calculate_GRC(global_model, client_models, client_losses):
    """
    Calculate GRC scores for clients based on parameter differences and losses.
    """
    # 1. Compute parameter differences between global and client models.
    param_diffs = []
    for model in client_models:
        diff = 0.0
        for g_param, l_param in zip(global_model.parameters(), model.parameters()):
            diff += torch.norm(g_param - l_param).item()
        param_diffs.append(diff)
    # 2. Map losses and parameter differences
    def map_sequence(sequence):
        max_val = max(sequence)
        min_val = min(sequence)
        return [(max_val - x) / (max_val + min_val) for x in sequence]  # Inverse relation

    mapped_losses = map_sequence(client_losses)
    mapped_diffs = map_sequence(param_diffs)
    
    ref_val = 1.0  # Reference ideal value for both metrics
    all_deltas = []
    for loss, diff in zip(mapped_losses, mapped_diffs):
        all_deltas.append(abs(loss - ref_val))
        all_deltas.append(abs(diff - ref_val))
    max_delta = max(all_deltas)
    min_delta = min(all_deltas)
    
    grc_losses = []
    grc_diffs = []
    for loss, diff in zip(mapped_losses, mapped_diffs):
        delta_loss = abs(loss - ref_val)
        delta_diff = abs(diff - ref_val)
        grc_loss = (min_delta + 0.5 * max_delta) / (delta_loss + 0.5 * max_delta)
        grc_diff = (min_delta + 0.5 * max_delta) / (delta_diff + 0.5 * max_delta)
        grc_losses.append(grc_loss)
        grc_diffs.append(grc_diff)
    
    grc_losses = np.array(grc_losses)
    grc_diffs = np.array(grc_diffs)
    # 3. Compute entropy weights based on the mapped metrics
    grc_metrics = np.vstack([mapped_losses, mapped_diffs])
    weights = entropy_weight(grc_metrics)
    # 4. Final weighted GRC score (using multiplication)
    weighted_score = grc_losses * weights[0] + grc_diffs * weights[1]
    
    return weighted_score, weights

def select_clients(client_loaders, num_select, global_model):
    """
    Select clients using only the GRA (GRC) strategy.
    Trains a local model on each client for 1 epoch, evaluates loss, calculates GRC scores,
    and then selects the top 'num_select' clients with the highest GRC scores.
    """
    client_models = []
    client_losses = []
    for client_id, client_loader in client_loaders.items():
        local_model = MLPModel()
        local_model.load_state_dict(global_model.state_dict())  # Sync with global model
        local_train(local_model, client_loader, epochs=1, lr=0.01)
        client_models.append(local_model)
        loss, _ = evaluate(local_model, client_loader)
        client_losses.append(loss)
    # Calculate GRC scores
    grc_scores, _ = calculate_GRC(global_model, client_models, client_losses)
    # Pair each client ID with its GRC score and sort in descending order (higher is better)
    client_grc_pairs = list(zip(client_loaders.keys(), grc_scores))
    client_grc_pairs.sort(key=lambda x: x[1], reverse=True)
    selected = [client_id for client_id, _ in client_grc_pairs[:num_select]]
    return selected

# -------------------------------
# Main Federated Learning Process (Using GRA Selection Only)
# -------------------------------
def main():
    # Set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Load MNIST dataset and split into 10 client subsets
    train_data, test_data = load_mnist_data()
    client_datasets, client_data_sizes = split_data_by_label(train_data, num_clients=10)

    # Create DataLoaders for each client and the test set
    client_loaders = {client_id: data.DataLoader(dataset, batch_size=32, shuffle=True)
                      for client_id, dataset in client_datasets.items()}
    test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize global model (using plain MLPModel)
    global_model = MLPModel()

    rounds = 100  # Total federated learning rounds
    num_selected_clients = 2  # From 10 clients, select 2 in each round
    
    # Initialize communication counters for each client
    communication_counts = {client_id: {'send': 0, 'receive': 0, 'full_round': 0}
                            for client_id in client_loaders.keys()}
    
    global_accuracies = []  # Track test accuracy for each round
    total_communication_counts = []  # Cumulative communication counts per round
    
    # CSV file to store round data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_data_{timestamp}.csv"
    csv_data = []
    
    for r in range(rounds):
        print(f"\n🔄 Round {r+1}")

        # Select clients using only the GRA (GRC) strategy
        selected_clients = select_clients(client_loaders, num_select=num_selected_clients, global_model=global_model)
        print(f"Selected clients: {selected_clients}")
        
        # Record that selected clients have received the global model
        update_communication_counts(communication_counts, selected_clients, "receive")

        client_state_dicts = []
        # Perform local training for each selected client
        for client_id in selected_clients:
            client_loader = client_loaders[client_id]
            local_model = MLPModel()
            local_model.load_state_dict(global_model.state_dict())
            local_state = local_train(local_model, client_loader, epochs=1, lr=0.1)
            client_state_dicts.append((client_id, local_state))
            update_communication_counts(communication_counts, [client_id], "send")
            
            param_mean = {name: param.mean().item() for name, param in local_model.named_parameters()}
            print(f"✅ Client {client_id} training complete | Sample count: {sum(client_data_sizes[client_id].values())}")
            print(f"📌 Client {client_id} model parameter mean: {param_mean}")
        
        # Compute round communication counts
        total_send = sum(communication_counts[c]['send'] - (communication_counts[c]['full_round'] - 1)
                         for c in selected_clients)
        total_receive = sum(communication_counts[c]['receive'] - (communication_counts[c]['full_round'] - 1)
                            for c in selected_clients)
        total_comm = total_send + total_receive
        if total_communication_counts:
            total_comm += total_communication_counts[-1]
        total_communication_counts.append(total_comm)
        
        # Aggregate client models via FedAvg
        global_model = fed_avg(global_model, client_state_dicts, client_data_sizes)
        
        # Evaluate the updated global model on the test set
        loss, accuracy = evaluate(global_model, test_loader)
        global_accuracies.append(accuracy)
        print(f"📊 Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.2f}%")
        
        csv_data.append([r+1, accuracy, total_comm, ",".join(map(str, selected_clients))])
        df = pd.DataFrame(csv_data, columns=["Round", "Accuracy", "Total Communication", "Selected Clients"])
        df.to_csv(csv_filename, index=False)
    
    final_loss, final_accuracy = evaluate(global_model, test_loader)
    print(f"\n🎯 Final Test Loss: {final_loss:.4f}")
    print(f"🎯 Final Test Accuracy: {final_accuracy:.2f}%")
    
    print("\nClient Communication Statistics:")
    for client_id, counts in communication_counts.items():
        print(f"Client {client_id}: Sent {counts['send']} times, Received {counts['receive']} times, Full rounds {counts['full_round']} times")
    
    # Plot test accuracy over rounds
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, rounds+1), global_accuracies, marker='o', linestyle='-', color='b', label="Test Accuracy")
    plt.xlabel("Federated Learning Rounds")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Over Federated Learning Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot test accuracy vs. total communication counts
    plt.figure(figsize=(8, 5))
    plt.plot(total_communication_counts, global_accuracies, marker='s', linestyle='-', color='r',
             label="Test Accuracy vs. Communication")
    plt.xlabel("Total Communication Count per Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy vs. Total Communication")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
