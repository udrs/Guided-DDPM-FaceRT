import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
# 定义LSTM模型

from torch.utils.data import Dataset


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x)
        # output: (batch_size, seq_len, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        # c_n: (num_layers, batch_size, hidden_size)
        out = self.fc(output[:, -1, :]) # Use the last output for prediction
        return out


# 超参数
input_size = 128
sequence_length = 9
hidden_size = 128
num_layers = 2
output_size = 18
batch_size = 128
learning_rate = 0.01
num_epochs = 100

# 设备配置
device = torch.device('cpu')

# 加载数据集
train_x = np.load("D:/我超，原/Baseline-with-HAR-datasets-main/Pre-processing/train_x.npy")
test_x = np.load("D:/我超，原/Baseline-with-HAR-datasets-main/Pre-processing/test_x.npy")
train_y = np.load("D:/我超，原/Baseline-with-HAR-datasets-main/Pre-processing/train_y.npy")
test_y = np.load("D:/我超，原/Baseline-with-HAR-datasets-main/Pre-processing/test_y.npy")

train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
train_loader = ((x.float(), y) for x, y in train_loader)
test_loader = ((x.float(), y) for x, y in test_loader)

data, label = next(iter(test_loader))

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''
# 训练模型
for epoch in range(num_epochs):

    train_correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.long().to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        print(total)
        train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%')

# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'测试集准确率: {100 * correct / total:.2f}%')
'''
sequence_dim = 9
input_dim = 128

loss_list = []
accuracy_list = []
iteration_list = []

iter = 0
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(train_loader):
        model.train()  # 声明训练
        # 一个batch的数据转换为RNN的输入维度
        data = data.view(-1, sequence_dim, input_dim).requires_grad_()
        label = label
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        # data = data[:, np.newaxis, :].squeeze()
        #print(data.shape)
        outputs = model(data)
        # 计算损失
        loss = criterion(outputs, label.long())
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计数器自动加1
        iter += 1
        # 模型验证
        if iter % 10 == 0:
            model.eval()  # 声明
            # 计算验证的accuracy
            correct = 0.0
            total = 0.0
            # 迭代测试集，获取数据、预测
            for data, label in test_loader:

                data = data.view(-1, sequence_dim, input_dim)

                # 模型预测
                outputs = model(data)
                # 获取预测概率最大值的下标
                predict = torch.max(outputs.data, 1)[1]
                # 统计测试集的大小
                total += label.size(0)
                # 统计判断/预测正确的数量
                correct += (predict == label).sum()
            # 计算
            if total > 0:
                accuracy = correct / total * 100
            else:
                accuracy = 0

            # 保存accuracy, loss, iteration
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            # 打印信息
            print("loop : {}, Loss : {}, Accuracy : {}".format(iter, loss.item(), accuracy))


plt.plot(iteration_list, accuracy_list, color='b')
plt.xlabel('Number of Iteration')
plt.ylabel('Accuracy')
plt.title('LSTM')
plt.show()