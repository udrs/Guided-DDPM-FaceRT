import pandas as pd
import numpy as np
from utils.Preprocessing import *
import torch
import torchvision

# model = torchvision.models.resnet18(pretrained=True)
# model = torchvision.models.resnet18(pretrained=False, NUM_CLASSES=10)


# number resnet18 分类num classes
# 全连接层
#model.fc 
# a = torch.rand ((4,3,224,224))
# b = model(a)
# print(b.shape)

# resnet sequence 文本分类模型 LSTM

# 序列数据 time sequence data

def load_data(file_path):
    data = pd.read_csv(file_path, sep="\s+")
    return data

file_path = "D:/myHAR/myHAR/data/test/Inertial Signals/body_acc_x_test.txt"
data = pd.read_csv(file_path, sep="\s+").iloc[:,STARTPOINT:STARTPOINT+WINDOW_WIDTH]
print(data.shape)

file_path2 = "D:/myHAR/myHAR/data/test/y_test.txt"
data2 = pd.read_csv(file_path2, sep="\s+")
print(data2.shape)


# X = load_data("../data/train/X_train.txt")
# y = load_data("../data/train/y_train.txt")
# y = np.asarray(y.values)
# actionA = X.iloc[np.argwhere(y==5)[:,0]]
# print(len(actionA))

