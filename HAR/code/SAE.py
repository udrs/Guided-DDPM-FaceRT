
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
train_data = np.load("Train_train.npz")
test_data = np.load("Test_Exp_test.npz")

train_x = torch.tensor(train_data['arr_0'])
train_y = torch.tensor(train_data['arr_1'])
test_x = torch.tensor(test_data['arr_0'])
test_y = torch.tensor(test_data['arr_1'])
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

BATCH_SIZE = 128
EPOCHS = 100
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

data, label = next(iter(test_loader))
print(data.shape)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1152, 700)
        self.hidden = nn.Linear(700, 200)
        self.output = nn.Linear(200, 18)
        self.relu = nn.Sigmoid()
        self.sigmoid = nn.Softmax()
    def forward(self, input):
        x = self.layer1(input)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

model = Model()
print(model)
loss_fn = nn.CrossEntropyLoss()
lr = 0.01
iter = 0
loss_list = []
accuracy_list = []
iteration_list = []
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
for epoch in range(50):
    for i, (data, label) in enumerate(train_loader):
        data = data.view(-1, 1152)
        output = model(data)
        loss = loss_fn(output, label.long())
        loss.backward()
        optimizer.step()
        print(loss)
    #with torch.no_grad:
    #    print("loss: ", loss)

        if iter % 10 == 0:
            model.eval()  # 声明
            # 计算验证的accuracy
            correct = 0.0
            total = 0.0
            # 迭代测试集，获取数据、预测
            for data, label in test_loader:

                data = data.view(-1, 1152)

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

plt.plot(iteration_list, accuracy_list, color='b')
plt.xlabel('Number of Iteration')
plt.ylabel('Accuracy')
plt.title('LSTM')
plt.show()