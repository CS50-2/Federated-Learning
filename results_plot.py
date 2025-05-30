import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def flatten_scalar(x):
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x
def prepend_zero_sum(time_list, acc_list):
    acc_list = list(acc_list)  # 把 pandas.Series 转成 list
    min_len = min(len(time_list), len(acc_list))
    x = [0] + list(np.cumsum(time_list[:min_len]))
    y = [0] + acc_list[:min_len]
    return x, y

def prepend_zero(time_list, acc_list):
    acc_list = list(acc_list)  # 把 pandas.Series 转成 list
    min_len = min(len(time_list), len(acc_list))
    x = [0] + time_list[:min_len]
    y = [0] + acc_list[:min_len]
    return x, y

def main():
    load_dir = "federated_results"

    # 定义通用读取函数
    def load_metric(name):
        path = os.path.join(load_dir, f"{name}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)[name].tolist()
        else:
            print(f"Warning: {name}.csv not found!")
            return []

    # 准备读取变量
    acc_grc = load_metric("acc_grc")
    acc_grc_smooth = load_metric("acc_grc_smooth")
    comm_grc = load_metric("comm_grc")
    flops_grc = load_metric("flops_grc")
    time_rounds_grc = load_metric("time_rounds_grc")
    time_grc = load_metric("time_grc")

    acc_grc_freeze_1 = load_metric("acc_grc_freeze_1")
    acc_grc_freeze_1_smooth = load_metric("acc_grc_freeze_1_smooth")
    comm_grc_freeze_1 = load_metric("comm_grc_freeze_1")
    flops_grc_freeze_1 = load_metric("flops_grc_freeze_1")
    time_rounds_grc_freeze_1 = load_metric("time_rounds_grc_freeze_1")
    time_grc_freeze_1 = load_metric("time_grc_freeze_1")

    acc_grc_freeze_2 = load_metric("acc_grc_freeze_2")
    acc_grc_freeze_2_smooth = load_metric("acc_grc_freeze_2_smooth")
    comm_grc_freeze_2 = load_metric("comm_grc_freeze_2")
    flops_grc_freeze_2 = load_metric("flops_grc_freeze_2")
    time_rounds_grc_freeze_2 = load_metric("time_rounds_grc_freeze_2")
    time_grc_freeze_2 = load_metric("time_grc_freeze_2")

    acc_grc_prune_30 = load_metric("acc_grc_prune_30")
    acc_grc_prune_30_smooth = load_metric("acc_grc_prune_30_smooth")
    comm_grc_prune_30 = load_metric("comm_grc_prune_30")
    flops_grc_prune_30 = load_metric("flops_grc_prune_30")
    time_rounds_grc_prune_30 = load_metric("time_rounds_grc_prune_30")
    time_grc_prune_30 = load_metric("time_grc_prune_30")

    acc_grc_prune_60 = load_metric("acc_grc_prune_60")
    acc_grc_prune_60_smooth = load_metric("acc_grc_prune_60_smooth")
    comm_grc_prune_60 = load_metric("comm_grc_prune_60")
    flops_grc_prune_60 = load_metric("flops_grc_prune_60")
    time_rounds_grc_prune_60 = load_metric("time_rounds_grc_prune_60")
    time_grc_prune_60 = load_metric("time_grc_prune_60")


    labels_grc = [
        "GRA Only",
        "GRA + Freeze 1 Layer",
        "GRA + Freeze 2 Layers",
        "GRA + Prune 30%",
        "GRA + Prune 60%",
    ]

    flops_values_grc = [flatten_scalar(v) for v in [
        flops_grc,
        flops_grc_freeze_1,
        flops_grc_freeze_2,
        flops_grc_prune_30,
        flops_grc_prune_60,
    ]]

    time_values_rounds_grc = [flatten_scalar(sum(v)) for v in [
        time_rounds_grc,
        time_rounds_grc_freeze_1,
        time_rounds_grc_freeze_2,
        time_rounds_grc_prune_30,
        time_rounds_grc_prune_60,
    ]]

    time_values_grc = [flatten_scalar(v[-1]) for v in [
        time_grc,
        time_grc_freeze_1,
        time_grc_freeze_2,
        time_grc_prune_30,
        time_grc_prune_60,
    ]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    bar_colors = ['skyblue', 'lightgreen','orange']
    metrics = ['FLOPs (TFLOPs)', 'Rounds Time (s)', 'Total Time (s)']
    titles = ['FLOPs Comparison', 'Rounds Time Comparison','Total Time Comparison']

    grc_values = [flops_values_grc,time_values_rounds_grc ,time_values_grc]

    # 绘制 GRC-based (下排)
    for i in range(3):
        bars = axes[i].bar(labels_grc, grc_values[i], color=bar_colors[i])
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.4f}", ha='center', va='bottom')
        axes[i].set_title(titles[i] + " (GRA-based)")
        axes[i].set_ylabel(metrics[i])
        axes[i].tick_params(axis='x', rotation=20)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 16))

    # ② GRC-based Accuracy Over Rounds
    axes[0, 0].plot(acc_grc, label="GRA Only", marker='o')
    axes[0, 0].plot(acc_grc_freeze_1, label="GRA + Freeze 1 Layer", marker='^')
    axes[0, 0].plot(acc_grc_freeze_2, label="GRA + Freeze 2 Layers", marker='v')
    axes[0, 0].plot(acc_grc_prune_30, label="GRA + Prune 30%", marker='s')
    axes[0, 0].plot(acc_grc_prune_60, label="GRA + Prune 60%", marker='D')
    axes[0, 0].set_title("Accuracy Over Rounds (GRA)")
    axes[0, 0].set_xlabel("Rounds")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # ④ GRC-based Accuracy vs Communication
    axes[0, 1].plot(comm_grc, acc_grc, label="GRA Only", marker='o')
    axes[0, 1].plot(comm_grc_freeze_1, acc_grc_freeze_1, label="GRA + Freeze 1 Layer", marker='^')
    axes[0, 1].plot(comm_grc_freeze_2, acc_grc_freeze_2, label="GRA + Freeze 2 Layers", marker='v')
    axes[0, 1].plot(comm_grc_prune_30, acc_grc_prune_30, label="GRA + Prune 30%", marker='s')
    axes[0, 1].plot(comm_grc_prune_60, acc_grc_prune_60, label="GRA + Prune 60%", marker='D')
    axes[0, 1].set_title("Accuracy vs Communication (GRA)")
    axes[0, 1].set_xlabel("Comm Count")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(acc_grc, label="GRA Only", linestyle='--' ,alpha=0.6)
    axes[1, 0].plot(acc_grc_smooth, label="GRA Only (Smoothed)")
    axes[1, 0].fill_between(range(len(acc_grc_smooth)), acc_grc_smooth, color='blue', alpha=0.2)
    axes[1, 0].plot(acc_grc_freeze_1, label="GRA + Freeze 1 Layer", linestyle='--' ,alpha=0.6)
    axes[1, 0].plot(acc_grc_freeze_1_smooth, label="GRA + Freeze 1 Layer (Smoothed)")
    axes[1, 0].fill_between(range(len(acc_grc_freeze_1_smooth)), acc_grc_freeze_1_smooth, color='orange', alpha=0.2)
    axes[1, 0].set_title("Accuracy Over Rounds (GRA)")
    axes[1, 0].set_xlabel("Rounds")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(acc_grc, label="GRA Only", linestyle='--' ,alpha=0.6)
    axes[1, 1].plot(acc_grc_smooth, label="GRA Only (Smoothed)")
    axes[1, 1].fill_between(range(len(acc_grc_smooth)), acc_grc_smooth, color='blue', alpha=0.2)
    axes[1, 1].plot(acc_grc_prune_30, label="GRA + Prune 30%", linestyle='--' ,alpha=0.6)
    axes[1, 1].plot(acc_grc_prune_30_smooth, label="GRA + Prune 30% (Smoothed)")
    axes[1, 1].fill_between(range(len(acc_grc_prune_30_smooth)), acc_grc_prune_30_smooth, color='red', alpha=0.2)
    axes[1, 1].set_title("Accuracy Over Rounds (GRA)")
    axes[1, 1].set_xlabel("Rounds")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[2, 0].plot(acc_grc_freeze_1, label="GRA + Freeze 1 Layer", linestyle='--' ,alpha=0.6)
    axes[2, 0].plot(acc_grc_freeze_1_smooth, label="GRA + Freeze 1 Layer (Smoothed)")
    axes[2, 0].fill_between(range(len(acc_grc_freeze_1_smooth)), acc_grc_freeze_1_smooth, color='orange', alpha=0.2)
    axes[2, 0].plot(acc_grc_freeze_2, label="GRA + Freeze 2 Layers", linestyle='--' ,alpha=0.6)
    axes[2, 0].plot(acc_grc_freeze_2_smooth, label="GRA + Freeze 2 Layer (Smoothed)")
    axes[2, 0].fill_between(range(len(acc_grc_freeze_2_smooth)), acc_grc_freeze_2_smooth, color='green', alpha=0.2)
    axes[2, 0].set_title("Accuracy Over Rounds (GRA)")
    axes[2, 0].set_xlabel("Rounds")
    axes[2, 0].set_ylabel("Accuracy (%)")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(acc_grc_prune_30, label="GRA + Prune 30%", linestyle='--' ,alpha=0.6)
    axes[2, 1].plot(acc_grc_prune_30_smooth, label="GRA + Prune 30% (Smoothed)")
    axes[2, 1].fill_between(range(len(acc_grc_prune_30_smooth)), acc_grc_prune_30_smooth, color='red', alpha=0.2)
    axes[2, 1].plot(acc_grc_prune_60, label="GRA + Prune 60%", linestyle='--' ,alpha=0.6)
    axes[2, 1].plot(acc_grc_prune_60_smooth, label="GRA + Prune 60% (Smoothed)")
    axes[2, 1].fill_between(range(len(acc_grc_prune_60_smooth)), acc_grc_prune_60_smooth, color='purple', alpha=0.2)
    axes[2, 1].set_title("Accuracy Over Rounds (GRA)")
    axes[2, 1].set_xlabel("Rounds")
    axes[2, 1].set_ylabel("Accuracy (%)")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # 累积时间 + 加0点
    time_total_1, acc_1 = prepend_zero(time_grc, acc_grc_smooth)
    time_total_2, acc_2 = prepend_zero(time_grc_freeze_1, acc_grc_freeze_1_smooth)
    time_total_3, acc_3 = prepend_zero(time_grc_freeze_2, acc_grc_freeze_2_smooth)
    time_total_4, acc_4 = prepend_zero(time_grc_prune_30, acc_grc_prune_30_smooth)
    time_total_5, acc_5 = prepend_zero(time_grc_prune_60, acc_grc_prune_60_smooth)

    time_round_1, _ = prepend_zero_sum(time_rounds_grc, acc_grc_smooth)
    time_round_2, _ = prepend_zero_sum(time_rounds_grc_freeze_1, acc_grc_freeze_1_smooth)
    time_round_3, _ = prepend_zero_sum(time_rounds_grc_freeze_2, acc_grc_freeze_2_smooth)
    time_round_4, _ = prepend_zero_sum(time_rounds_grc_prune_30, acc_grc_prune_30_smooth)
    time_round_5, _ = prepend_zero_sum(time_rounds_grc_prune_60, acc_grc_prune_60_smooth)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 配色（可自定义）
    colors = {
        'GRA Only': 'tab:blue',
        'Freeze 1': 'tab:orange',
        'Freeze 2': 'tab:green',
        'Prune 30': 'tab:red',
        'Prune 60': 'tab:purple'
    }

    # 左图：Total Training Time
    axes[0].plot(time_total_1, acc_1, label='GRA Only', color=colors['GRA Only'])
    axes[0].fill_between(time_total_1, acc_1, alpha=0.2, color=colors['GRA Only'])

    axes[0].plot(time_total_2, acc_2, label='GRA + Freeze 1 Layer', color=colors['Freeze 1'])
    axes[0].fill_between(time_total_2, acc_2, alpha=0.2, color=colors['Freeze 1'])

    axes[0].plot(time_total_3, acc_3, label='GRA + Freeze 2 Layers', color=colors['Freeze 2'])
    axes[0].fill_between(time_total_3, acc_3, alpha=0.2, color=colors['Freeze 2'])

    axes[0].plot(time_total_4, acc_4, label='GRA + Prune 30%', color=colors['Prune 30'])
    axes[0].fill_between(time_total_4, acc_4, alpha=0.2, color=colors['Prune 30'])

    axes[0].plot(time_total_5, acc_5, label='GRA + Prune 60%', color=colors['Prune 60'])
    axes[0].fill_between(time_total_5, acc_5, alpha=0.2, color=colors['Prune 60'])

    axes[0].set_title("Accuracy vs Total Training Time")
    axes[0].set_xlabel("Cumulative Total Time (s)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True)

    # 右图：Round Execution Time
    axes[1].plot(time_round_1, acc_1, label='GRA Only', color=colors['GRA Only'])
    axes[1].fill_between(time_round_1, acc_1, alpha=0.2, color=colors['GRA Only'])

    axes[1].plot(time_round_2, acc_2, label='GRA + Freeze 1 Layer', color=colors['Freeze 1'])
    axes[1].fill_between(time_round_2, acc_2, alpha=0.2, color=colors['Freeze 1'])

    axes[1].plot(time_round_3, acc_3, label='GRA + Freeze 2 Layers', color=colors['Freeze 2'])
    axes[1].fill_between(time_round_3, acc_3, alpha=0.2, color=colors['Freeze 2'])

    axes[1].plot(time_round_4, acc_4, label='GRA + Prune 30%', color=colors['Prune 30'])
    axes[1].fill_between(time_round_4, acc_4, alpha=0.2, color=colors['Prune 30'])

    axes[1].plot(time_round_5, acc_5, label='GRA + Prune 60%', color=colors['Prune 60'])
    axes[1].fill_between(time_round_5, acc_5, alpha=0.2, color=colors['Prune 60'])

    axes[1].set_title("Accuracy vs Round Execution Time")
    axes[1].set_xlabel("Cumulative Round Time (s)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()