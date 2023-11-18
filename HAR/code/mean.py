import os
import time
import glob
import torch

from utils.constants import *
from utils.NeuralNetwork import *
from utils.Plot import *
from utils.basic import *

prepare_data = Preprocessing() 
data_X, data_y = prepare_data.trans("train", True)

print("data_X:{}".format(data_X))
print("data_y:{}".format(data_y))
