import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim

train_x = np.load("/home/wendi/Pre-processing/train_x.npy")
test_x = np.load("/home/wendi/Pre-processing/test_x.npy")
train_y = np.load("/home/wendi/Pre-processing/train_y.npy")
test_y = np.load("/home/wendi/Pre-processing/test_y.npy")


train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).long()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).long()
print(train_x.shape)

# 创建TensorDataset，包含时序数据和特征数据
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

BATCH_SIZE = 128
EPOCHS = 60

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class LSTM_CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(LSTM_CNN, self).__init__()

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # Calculate the input size for the fully connected layer
        fc_input_size = 32 * (((9 - 3 + 1) // 2))
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        # CNN layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Flatten and Fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


model = LSTM_CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and testing the model
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Plotting the accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()