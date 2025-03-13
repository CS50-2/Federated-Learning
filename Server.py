import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from threading import Thread

# 服务器参数
SERVER_HOST = '127.0.0.1'  # 服务器 IP
SERVER_PORT = 12345        # 服务器端口
NUM_CLIENTS = 5            # 客户端数量
BUFFER_SIZE = 4096         # Socket 传输数据缓冲区大小

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 联邦平均 (FedAvg)
def fed_avg(client_models):
    global_model = CNNModel().state_dict()
    total_clients = len(client_models)
    for key in global_model.keys():
        global_model[key] = sum(client_model[key] for client_model in client_models) / total_clients
    return global_model

def receive_full_data(client_socket):
    """ 从客户端接收完整的 pickle 数据 """
    data_buffer = b""  # 用于存储完整的数据
    while True:
        packet = client_socket.recv(4096)  # 逐块接收数据
        if not packet:
            break
        data_buffer += packet  # 拼接数据
    return pickle.loads(data_buffer)  # 反序列化数据
# 处理客户端连接
def handle_client(client_socket, client_models):
    # 接收客户端模型参数
    data = client_socket.recv(BUFFER_SIZE)
    client_model = receive_full_data(client_socket)  # 反序列化模型参数
    client_models.append(client_model)

    print(f"📥 收到客户端 {len(client_models)} 的模型参数")

    # 等待所有客户端传输完毕
    if len(client_models) == NUM_CLIENTS:
        print("🔄 开始全局模型聚合 (FedAvg)")
        global_model = fed_avg(client_models)  # 执行联邦平均
        client_models.clear()  # 清空客户端模型缓存

        # 发送更新后的全局模型
        serialized_model = pickle.dumps(global_model)
        for _ in range(NUM_CLIENTS):
            client_socket.send(serialized_model)

        print("📤 发送更新后的全局模型给所有客户端")

    client_socket.close()

# 启动服务器
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(NUM_CLIENTS)
    print(f"🚀 服务器启动，监听 {SERVER_HOST}:{SERVER_PORT}")

    client_models = []

    while True:
        client_socket, addr = server_socket.accept()
        print(f"✅ 客户端 {addr} 连接成功")
        client_thread = Thread(target=handle_client, args=(client_socket, client_models))
        client_thread.start()

if __name__ == "__main__":
    start_server()