import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# 加载特征名称
with open("UCI HAR Dataset/features.txt") as f:
    features = [line.strip() for line in f.readlines()]

# 加载活动类别标签
activity_labels = pd.read_csv("UCI HAR Dataset/activity_labels.txt", sep=" ", header=None, names=["label", "activity"])

# 加载训练数据
train_features = pd.read_csv("UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, header=None, names=features)
#y_train = pd.read_csv("C:/Users/97233/Desktop/UCI HAR Dataset/y_train.txt", header=None, names=["label"])

# 加载测试数据
test_features = pd.read_csv("UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None, names=features)
#y_test = pd.read_csv("C:/Users/97233/Desktop/UCI HAR Dataset", header=None, names=["label"])

# 将标签数据与活动类别标签进行合并，以获取活动名称
#y_train = y_train.merge(activity_labels, on="label")
#y_test = y_test.merge(activity_labels, on="label")

train_x = np.load("/home/wendi/Pre-processing/train_x.npy")
test_x = np.load("/home/wendi/Pre-processing/test_x.npy")
train_y = np.load("/home/wendi/Pre-processing/train_y.npy")
test_y = np.load("/home/wendi/Pre-processing/test_y.npy")

train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).long()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).long()
train_features = torch.tensor(train_features.values).float()
test_features = torch.tensor(test_features.values).float()
print(train_x.shape)
print(train_features.shape)
# 创建TensorDataset，包含时序数据和特征数据
train_dataset = TensorDataset(train_x, train_features, train_y)
test_dataset = TensorDataset(test_x, test_features, test_y)

BATCH_SIZE = 128
EPOCHS = 20

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

##data, label = next(iter(test_loader))?
#print(data.shape)
#print(label.shape)
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

class CombinedModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_layer_dim, lstm_output_dim,
                 fc_input_dim, fc_output_dim, combined_output_dim, num_classes):
        super(CombinedModel, self).__init__()

        # LSTM module
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, lstm_layer_dim, batch_first=True)
        self.fc_lstm = nn.Linear(lstm_hidden_dim, lstm_output_dim)

        # Fully connected module
        self.fc_feature = nn.Linear(fc_input_dim, fc_output_dim)

        # Concatenate, BatchNorm and Dropout
        self.fc_combined = nn.Linear(combined_output_dim, num_classes)
        self.bn = nn.BatchNorm1d(combined_output_dim)
        self.dropout = nn.Dropout(0.5)
        self.dropout_lstm = nn.Dropout(0.5)

    def forward(self, x_lstm, x_fc):
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x_lstm)
        output = self.dropout_lstm(output[:, -1, :])  # Apply dropout to LSTM output
        out_lstm = self.fc_lstm(output)

        # Fully connected forward
        out_fc = self.fc_feature(x_fc)

        # Concatenate LSTM and fully connected outputs
        out_combined = torch.cat((out_lstm, out_fc), dim=1)

        # BatchNorm and Dropout
        out_bn = self.bn(out_combined)
        out_dropout = self.dropout(out_bn)

        # Output layer
        out = self.fc_combined(out_dropout)

        # Softmax
        return nn.functional.softmax(out, dim=1)

# LSTM 相关参数
lstm_input_dim = 128
lstm_hidden_dim = 256
lstm_layer_dim = 2
lstm_output_dim = 128

# 全连接层相关参数
fc_input_dim = 561
fc_output_dim = 128

# 合并后输出层的参数
combined_output_dim = lstm_output_dim + fc_output_dim

# 创建 CombinedModel 实例
num_classes = 7
model = CombinedModel(lstm_input_dim, lstm_hidden_dim, lstm_layer_dim, lstm_output_dim,
                      fc_input_dim, fc_output_dim, combined_output_dim, num_classes)



criterion = nn.CrossEntropyLoss()
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

length = len(list(model.parameters()))
for i in range(length):
    print ('参数：%d'%(i+1))
    print(list(model.parameters())[i].size())

sequence_dim = 9
loss_list = []
accuracy_list = []
iteration_list = []

iter = 0
best_accuracy = 0.0
best_model_state = model.state_dict().copy()

for epoch in range(EPOCHS):
    for i, (data_lstm, data_fc, label) in enumerate(train_loader):
        model.train()  # 声明训练

        data_lstm, data_fc, label = data_lstm.to(device), data_fc.to(device), label.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        # data = data[:, np.newaxis, :].squeeze()
        #print(data.shape)
        outputs = model(data_lstm, data_fc)
        # 计算损失
        loss = criterion(outputs, label.long())
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计数器自动加1
        iter += 1

        ##

        # 模型验证
        if iter % 10 == 0:
            model.eval()  # 声明
            # 计算验证的accuracy
            correct = 0.0
            total = 0.0

            model.load_state_dict(best_model_state)
            # 迭代测试集，获取数据、预测
            for data_lstm, data_fc, label in test_loader:

                data_lstm, data_fc, label = data_lstm.to(device), data_fc.to(device), label.to(device)

                # 模型预测
                outputs = model(data_lstm, data_fc)
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
            ##
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict()
            # 打印信息
            print("loop : {}, Loss : {}, Accuracy : {}".format(iter, loss.item(), accuracy))

plt.plot(iteration_list, accuracy_list, color='b')
plt.xlabel('Number of Iteration')
plt.ylabel('Accuracy')
plt.title('LSTM')
plt.show()

output_list = []
prediction_list = []

output_list = []
prediction_list = []
true_labels = []

for lstm_data, feature_data, label in test_loader:
    outputs = model(lstm_data, feature_data)
    predict = torch.max(outputs.data, 1)[1]
    output_list.extend(outputs.detach().cpu().numpy())
    prediction_list.extend(predict.cpu().numpy())
    true_labels.extend(label.cpu().numpy())

result_df = pd.DataFrame(output_list, columns=["Output_{}".format(i) for i in range(num_classes)])
result_df["Prediction"] = prediction_list
result_df["True_Label"] = true_labels
result_df.to_csv("model_outputs_and_predictions.csv", index=False)

# After the test loop
true_labels = []
predicted_labels = []

for lstm_data, feature_data, label in test_loader:
    outputs = model(lstm_data, feature_data)
    predict = torch.max(outputs.data, 1)[1]
    predicted_labels.extend(predict.cpu().numpy())
    true_labels.extend(label.cpu().numpy())

# Calculate the F1 score for each class and average F1 score
f1_scores = f1_score(true_labels, predicted_labels, average=None)
average_f1_score = f1_score(true_labels, predicted_labels, average='weighted')

# Print the F1 scores for each class and the average F1 score
print("F1 Scores for Each Class:")
for i, f1 in enumerate(f1_scores):
    print(f"Class {i}: {f1}")

print(f"Average F1 Score: {average_f1_score}")

# Generate and print a classification report
class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:")
print(report)
