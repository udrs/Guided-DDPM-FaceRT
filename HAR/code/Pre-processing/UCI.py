import pandas as pd
import numpy as np

# 加载特征名称
with open("features.txt") as f:
    features = [line.strip() for line in f.readlines()]

# 加载活动类别标签
activity_labels = pd.read_csv("activity_labels.txt", sep=" ", header=None, names=["label", "activity"])

# 加载训练数据
X_train = pd.read_csv("train/X_train.txt", delim_whitespace=True, header=None, names=features)
y_train = pd.read_csv("train/y_train.txt", header=None, names=["label"])

# 加载测试数据
X_test = pd.read_csv("test/X_test.txt", delim_whitespace=True, header=None, names=features)
y_test = pd.read_csv("test/y_test.txt", header=None, names=["label"])

# 将标签数据与活动类别标签进行合并，以获取活动名称
y_train = y_train.merge(activity_labels, on="label")
y_test = y_test.merge(activity_labels, on="label")

print(X_train)
'''
np.save("train_y.npy", train_y)
np.save("test_y.npy", test_y)

np.save("train_x.npy", train_x.transpose(0, 2, 1))
np.save("test_x.npy", test_x.transpose(0, 2, 1))
'''
