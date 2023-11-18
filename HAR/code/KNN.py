import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import glob
import torch
import copy
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils.Preprocessing import *
from utils.constants import *
from utils.Plot import *
from utils.Preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

prepare_data = Preprocessing() 
data_X, data_y = prepare_data.trans('train', True)

data_X=data_X.view(-1,1152)

data_X=data_X.numpy()
data_y=data_y.numpy()

# data_X:(28194, 1152)
# data_Y:(28194,)

print("data_X:{}".format(data_X.shape))
print("data_Y:{}".format(data_y.shape))

prepare_data2 = Preprocessing()
test_x, test_y = prepare_data2.trans("test", True)

test_x=test_x.view(-1,1152)

test_x=test_x.numpy()
test_y=test_y.numpy()

print("test_x:{}".format(test_x.shape))
print("test_y:{}".format(test_y.shape))


# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# # Assign colum names to the dataset
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# # Read dataset to pandas dataframe
# dataset = pd.read_csv(url, names=names)

# dataset.head()

# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 4].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# # numpy type

# print(X_train.shape)  # (120,4)
# print(y_train.shape)  # (120,)
# print(X_test.shape)  # (30, 4)
# print(y_test.shape)  # (30,)


scaler = StandardScaler()
scaler.fit(data_X)

X_train = scaler.transform(data_X)
X_test = scaler.transform(test_x)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(data_X, data_y)

y_pred = classifier.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))