import numpy
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=9999)

# 禁用科学计数法
torch.set_printoptions(sci_mode=False, precision=3)

result_df = pd.read_csv("model_outputs_and_predictions.csv")

# 计算整体预测准确率
overall_accuracy = accuracy_score(result_df["True_Label"], result_df["Prediction"])
print("Overall Accuracy:", overall_accuracy)

# 计算每个类别的准确率
class_report = classification_report(result_df["True_Label"], result_df["Prediction"])
print("\nClass-wise Accuracy:")
print(class_report)

##
#real = np.load("y_test.npy")
#Y = np.load("train_y.npy")

data_frame = pd.read_csv("model_outputs_and_predictions.csv")
data_frame = data_frame.drop(columns=["Prediction", "True_Label"])
posterior = data_frame.to_numpy()
train_y = np.load("Train_train.npz")
Y = train_y['arr_1']
test_y = np.load("Test_Exp_test.npz")
real = test_y['arr_1']
print(real.shape)
real = real - 1
Y = Y - 1
posterior = posterior[:, 1:]
posterior = torch.tensor(posterior)
posterior = torch.softmax(posterior, dim=1)

# 读取CSV文件
dataframe = pd.read_csv("model_outputs_and_predictions.csv")

# 提取预测值列
predictions_column = dataframe["Prediction"]

# 将预测值列转换为NumPy数组
pred = predictions_column.to_numpy()
pred -= 1
print(pred.shape)
#pred = np.argmax(posterior, axis=1)

accuracy = accuracy_score(real, pred)
print("accuracy:", accuracy)


########################转换概率矩阵transition probability matrix#############################
transition_probability = torch.zeros(17, 17, dtype=torch.float64)

# given that Y is a sequence that ranges from 0 to 5, all integers, calculate the transition probability matrix for Y
for i in range(0, len(Y)-1):
    transition_probability[Y[i], Y[i+1]] = transition_probability[Y[i], Y[i+1]] + 1
transition_probability = transition_probability / transition_probability.sum()

# Give minor prob to any transition that has not been observed
transition_probability = transition_probability + 1e-3
transition_probability = transition_probability / transition_probability.sum()

print('transition probability matrix: \n', transition_probability)

########################先验分布p(z_t) prior distribution#############################
prior_distribution = torch.zeros(17)

for item in Y:
    prior_distribution[item] = prior_distribution[item] + 1
prior_distribution = prior_distribution / prior_distribution.sum()

print('prior distribution: \n', prior_distribution)

########################后验分布p(z_t|o_{t-m,t}) posterior distribution#############################
m_steps = 3
m_step_posterior = torch.zeros_like(posterior)

m_step_posterior[0, :] = posterior[0, :]

for i in range(1, len(m_step_posterior)):
    m_step_posterior[i, :] = transition_probability @ posterior[i - 1, :]
    m_step_posterior[i, :] = m_step_posterior[i, :] / m_step_posterior[i, :].sum()

    for j in range(1, m_steps):
        m_step_posterior[i, :] = transition_probability @ m_step_posterior[i, :]
        m_step_posterior[i, :] = m_step_posterior[i, :] / m_step_posterior[i, :].sum()


m_step_pred = np.argmax(m_step_posterior, axis=1)
m_step_accuracy = accuracy_score(real, m_step_pred)
print("m_step_accuracy:", m_step_accuracy)


########################最终分布r^i_t final distribution#############################
final_distribution = torch.zeros_like(posterior)

for i in range(len(final_distribution)):
    final_distribution[i, :] = m_step_posterior[i, :] * posterior[i, :] / prior_distribution
    final_distribution[i, :] = final_distribution[i, :] / final_distribution[i, :].sum()

final_pred = np.argmax(final_distribution, axis=1)
final_accuracy = accuracy_score(real, final_pred)

print("final_accuracy:", final_accuracy, 'm_steps:', m_steps)

'''
for i in range(0, len(final_distribution)):
    if pred[i] == 3:
        pred[i] == final_pred[i]
print(accuracy_score(real, pred))
'''