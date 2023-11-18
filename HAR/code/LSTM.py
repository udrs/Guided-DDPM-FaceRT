import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



train_x = np.load("/media/lscsc/nas/wendi/CodeFormerDDPM/HAR/code/Pre-processing/train_x.npy")
test_x = np.load("/media/lscsc/nas/wendi/CodeFormerDDPM/HAR/code/Pre-processing/test_x.npy")
train_y = np.load("/media/lscsc/nas/wendi/CodeFormerDDPM/HAR/code/Pre-processing/train_y.npy")
test_y = np.load("/media/lscsc/nas/wendi/CodeFormerDDPM/HAR/code/Pre-processing/test_y.npy")

train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).long()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).long()


train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

BATCH_SIZE = 128
EPOCHS = 10
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
data, label = next(iter(test_loader))


print(data.shape)
print(label.shape)
device = torch.device('cpu')
class RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)#nonlinearity='relu')

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        output, (h_n, c_n) = self.rnn(x)
        out = self.fc(output[:, -1, :])

        return out


input_dim = 128
hidden_dim = 256
layer_dim = 2
output_dim = 18

model = RNN_Model(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.CrossEntropyLoss()
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

length = len(list(model.parameters()))

sequence_dim = 9
loss_list = []
accuracy_list = []
iteration_list = []

output_list = []
prediction_list = []

iter = 0
best_accuracy = 0
best_model_state = model.state_dict().copy()

for epoch in range(EPOCHS):
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
        if torch.cuda.is_available():
            correct += (predict == label).sum()
        else:
            correct += (predict == label).sum()
    # 计算
        accuracy = correct / total * 100
        # 保存accuracy, loss, iteration
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)
        iteration_list.append(iter)
    # 打印信息

        print("loop : {}, Loss : {}, Accuracy : {}".format(iter, loss.item(), accuracy))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model.state_dict()

output_list = []
prediction_list = []
true_labels = []

model.load_state_dict(best_model_state)
for data, label in test_loader:
    data = data.view(-1, sequence_dim, input_dim)
    outputs = model(data)
    predict = torch.max(outputs.data, 1)[1]
    output_list.extend(outputs.detach().cpu().numpy())
    prediction_list.extend(predict.cpu().numpy())
    true_labels.extend(label.cpu().numpy())

plt.plot(iteration_list, accuracy_list, color='b')
plt.xlabel('Number of Iteration')
plt.ylabel('Accuracy')
plt.title('LSTM')
plt.savefig('loss_curve.png') 
plt.show()

result_df = pd.DataFrame(output_list, columns=["Output_{}".format(i) for i in range(output_dim)])
result_df["Prediction"] = prediction_list
result_df["True_Label"] = true_labels
result_df.to_csv("model_outputs_and_predictions.csv", index=False)
