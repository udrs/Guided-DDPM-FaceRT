#!/usr/bin/env python
# coding: utf-8

# In[94]:


from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, global_add_pool

train_x = np.load("/home/wendi/code/train_x.npy")
test_x = np.load("/home/wendi/code/test_x.npy")
train_y = np.load("/home/wendi/code/train_y.npy")
test_y = np.load("/home/wendi/code/test_y.npy")


# In[95]:


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# In[96]:


train_x = train_x.reshape(-1,9 * 128)
test_x = test_x.reshape(-1,9 * 128)


# In[97]:


train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).long()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).long()

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

BATCH_SIZE = 64
EPOCHS = 50

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

class GNNClassifier(nn.Module):
    def __init__(self, input_size= 9 * 128, hidden_size=32, num_classes=7):
        super(GNNClassifier, self).__init__()

        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GatedGraphConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
#         x = global_add_pool(x, batch)
        x = self.fc(x)
        return x
# 配置模型和优化器
model = GNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[98]:


print(model)


# In[100]:


# 训练模型
train_losses = []
test_losses = []
test_accuracies = []
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
num_nodes = len(train_x)
batch_size = 128
num_batches = num_nodes // batch_size
batch = torch.tensor([i for i in range(num_nodes)], dtype=torch.long)
#batch = (batch // batch_size).tolist()
batch = (torch.div(batch, batch_size, rounding_mode='floor')).tolist()

batch = torch.tensor(batch, dtype=torch.long)

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
#         data = data.transpose(1, 2)  # 调整数据维度，使之符合GNN的输入格式
        output = model(data, edge_index, batch)  # edge_index和batch表示图的边和节点信息
        #output = model(data, edge_index, batch)  # edge_index should be specific to each instance

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
#             data = data.transpose(1, 2)  # 调整数据维度，使之符合GNN的输入格式
            output = model(data, edge_index, batch)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, TestAccuracy: {test_accuracy:.2f}%')

plt.figure(figsize=(10, 5))
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

