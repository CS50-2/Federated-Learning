import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

# 服务器参数
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345
BUFFER_SIZE = 4096

# 载入 MNIST 数据集
def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    return train_data

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

# 本地训练
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

# 连接服务器并传输数据
def connect_to_server(local_model):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))

    # 发送本地训练的模型参数
    serialized_model = pickle.dumps(local_model)
    client_socket.send(serialized_model)

    print("📤 客户端发送本地模型参数至服务器")

    # 接收服务器返回的全局模型
    data = client_socket.recv(BUFFER_SIZE)
    global_model = pickle.loads(data)
    print("📥 客户端接收更新后的全局模型")

    client_socket.close()
    return global_model

def main():
    # 加载数据集并划分数据
    train_data = load_mnist_data()
    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)

    # 初始化本地模型
    local_model = CNNModel()

    # 进行本地训练
    local_model_params = local_train(local_model, train_loader, epochs=1, lr=0.01)

    # 连接服务器并传输本地模型
    global_model_params = connect_to_server(local_model_params)

    # 更新本地模型
    local_model.load_state_dict(global_model_params)

if __name__ == "__main__":
    main()