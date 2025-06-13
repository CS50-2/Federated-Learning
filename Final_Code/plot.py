import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Ensure output folder exists
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# File paths and method labels
files = {
    'GRA': 'GRA.csv',
    'LoRA': 'LoRA.csv',
    'NoRA': 'NoRA_0.6.csv',
    'SVDAB': 'SVDAB.csv'
}

# # File paths and method labels
# files = {
#     'GRA': 'GRA.csv',
#     'NoRA': 'NoRA_0.6.csv'
# }


# Set academic style
sns.set_theme(style="whitegrid", font="serif", font_scale=1.2)
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

# Define consistent color mapping across all plots
fixed_palette = {
    'GRA': '#1f77b4',     # blue
    'LoRA': '#2ca02c',    # green
    'NoRA': '#ff7f0e',   # orange
    'SVDAB': '#d62728'    # red 
}

# # Define consistent color mapping across all plots
# fixed_palette = {
#     'GRA': '#1f77b4',     # blue
#     'NoRA': '#ff7f0e'    
# }


# --- Line Plot: Round vs Accuracy (smoothed) ---
plt.figure(figsize=(10, 6))
for label, file in files.items():
    df = pd.read_csv(file).head(300)  # Only use the first 300 rounds
    df['smoothed_accuracy'] = df['Accuracy'].rolling(window=10, min_periods=1).mean()
    plt.plot(df['Round'], df['smoothed_accuracy'], label=label,
             color=fixed_palette[label], linewidth=2)
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Round vs Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "round_vs_accuracy.png"), dpi=300)

# --- Line Plot: Elapsed Time vs Accuracy with shaded region ---
plt.figure(figsize=(10, 6))
for label, file in files.items():
    df = pd.read_csv(file).head(300)
    
    # Rolling statistics
    df['smoothed_accuracy'] = df['Accuracy'].rolling(window=10, min_periods=1).mean()
    df['std_accuracy'] = df['Accuracy'].rolling(window=10, min_periods=1).std()

    # Plot line
    plt.plot(df['Elapsed Time (s)'], df['smoothed_accuracy'],
             label=label, color=fixed_palette[label], linewidth=2)

    # Fill area under the curve (line to x-axis)
    plt.fill_between(df['Elapsed Time (s)'],
                     df['smoothed_accuracy'],
                     color=fixed_palette[label],
                     alpha=0.2)  # Transparency

plt.xlabel('Elapsed Time (s)')
plt.ylabel('Accuracy')
plt.title('Elapsed Time vs Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "time_vs_accuracy_shaded.png"), dpi=300)
# plt.show()

# --- Extract cumulative values from CSVs ---
methods = []
param_usages = []      # Parameter usage in millions
flops_usages = []      # FLOPs in billions
total_times = []

for method, file in files.items():
    df = pd.read_csv(file).head(300)
    methods.append(method)
    param_usages.append(df['Parameter Usage'].iloc[-1] / 1e6)  # Convert to millions
    flops_usages.append(df['FLOPs'].iloc[-1] / 1e9)             # Convert to billions
    total_times.append(df['Elapsed Time (s)'].iloc[-1])  # Final accumulated time

# --- Bar Plot: Parameter Usage ---
plt.figure(figsize=(8, 6))
sns.barplot(x=methods, y=param_usages, palette=fixed_palette)
plt.title("Total Parameter Usage (300 Rounds)")
plt.ylabel("Parameters (Millions)")
plt.xlabel("Method")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_parameter_usage.png"), dpi=300)
# plt.show()

# --- Bar Plot: FLOPs Usage ---
plt.figure(figsize=(8, 6))
sns.barplot(x=methods, y=flops_usages, palette=fixed_palette)
plt.title("Total FLOPs")
plt.ylabel("FLOPs (Billions)")
plt.xlabel("Method")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_flops_usage.png"), dpi=300)
# plt.show()

# --- Bar Plot: Elapsed Time ---
plt.figure(figsize=(8, 6))
sns.barplot(x=methods, y=total_times, palette=fixed_palette)
plt.title("Total Elapsed Time")
plt.ylabel("Elapsed Time (seconds)")
plt.xlabel("Method")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_elapsed_time.png"), dpi=300)
# plt.show()

# --- Linear Plot: Parameters vs Accuracy ---
plt.figure(figsize=(10, 6))

for method, file in files.items():
    df = pd.read_csv(file).head(300)
    
    # Convert Parameter Usage to millions
    df['Parameter Usage (M)'] = df['Parameter Usage'] / 1e6
    
    # Apply smoothing to Accuracy
    df['Smoothed Accuracy'] = df['Accuracy'].rolling(window=10, min_periods=1).mean()
    
    # Plot line
    plt.plot(df['Parameter Usage (M)'], df['Smoothed Accuracy'],
             label=method, color=fixed_palette[method], linewidth=2)
    
    # Area fill under the curve
    plt.fill_between(df['Parameter Usage (M)'],
                     df['Smoothed Accuracy'],
                     color=fixed_palette[method],
                     alpha=0.2)

# Axis labels and title
plt.xlabel("Cumulative Parameter Usage (Millions)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Parameter Usage")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_vs_parameters.png"), dpi=300)
# plt.show()

# --- Linear Plot: FLOPs vs Accuracy ---
plt.figure(figsize=(10, 6))

for method, file in files.items():
    df = pd.read_csv(file).head(300)
    
    # Convert FLOPs to trillions (optional: use / 1e9 for billions)
    df['FLOPs (T)'] = df['FLOPs'] / 1e12
    
    # Apply smoothing to Accuracy
    df['Smoothed Accuracy'] = df['Accuracy'].rolling(window=10, min_periods=1).mean()
    
    # Plot line
    plt.plot(df['FLOPs (T)'], df['Smoothed Accuracy'],
             label=method, color=fixed_palette[method], linewidth=2)
    
    # Fill area between line and x-axis
    plt.fill_between(df['FLOPs (T)'],
                     df['Smoothed Accuracy'],
                     color=fixed_palette[method],
                     alpha=0.2)

# Axis labels and title
plt.xlabel("Cumulative FLOPs (Trillions)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs FLOPs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_vs_flops.png"), dpi=300)
# plt.show()