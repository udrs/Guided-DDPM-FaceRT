
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.constants import Conv1d_ft, Conv1d_st
from utils.NeuralNetwork import Conv1DNet
import scienceplots
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)



train_data = np.load("/home/wendi/code/Train_train.npz")
test_data = np.load("/home/wendi/code/Test_Exp_test.npz")
Xtr, ytr = train_data['arr_0'], train_data['arr_1']
Xte, yte = test_data['arr_0'] , test_data['arr_1']

torch.tensor(Xtr), torch.tensor(ytr)

train_dataset = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
test_dataset = TensorDataset(torch.tensor(Xte), torch.tensor(yte))

train_loader = DataLoader(train_dataset, batch_size=128, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=128, drop_last=True)


fc1_input_dim = Conv1d_ft[1] * (Conv1d_ft[1] - Conv1d_st[0] + 1) #CC
#net = Conv1DNet(fc1_input_dim).cuda()




loss_list1 = []
loss_list2 = []
loss_list3 = []
loss_list4 = []
loss_list5 = []

for run in range(5): 
    i = 1
    ##
    net = Conv1DNet(fc1_input_dim)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for _ in range(100):

        for i, (xbatch, ybatch) in enumerate(train_loader):
            xbatch, ybatch = xbatch.float(), ybatch.long()

            optimizer.zero_grad()

            ypred = net(xbatch)
            loss = criterion(ypred, ybatch)

            loss.backward()

            optimizer.step()
            
            
        with torch.no_grad():
            ypred_list = []
            ybatch_list = []
            loss_list = []
            ##
            running_loss = 0.0
            ##
            for i, (xbatch, ybatch) in enumerate(test_loader):
                xbatch, ybatch = xbatch.float(), ybatch.long()
                
                ypred = net(xbatch)
                loss = criterion(ypred, ybatch)
                
                ybatch_list.append(ybatch)
                ypred_list.append(ypred.max(dim=1)[1])
                loss_list.append(loss)
                ##
                running_loss += loss.item()
                
            epoch_loss = running_loss / len(train_loader)

            if run == 0:
                loss_list1.append(epoch_loss)
            elif run == 1:
                loss_list2.append(epoch_loss)
            elif run == 2:
                loss_list3.append(epoch_loss)
            elif run == 3:
                loss_list4.append(epoch_loss)
            else:
                loss_list5.append(epoch_loss)
            loss_test_all = torch.hstack(loss_list).cpu().numpy()
            ytest_pred = torch.hstack(ypred_list).cpu().numpy()
            ytest_real = torch.hstack(ybatch_list).cpu().numpy()
            
            acc = accuracy_score(ytest_pred, ytest_real)
            #print(f'After testing, the loss is {loss_test_all.mean()}')
            print("accuracy: ", acc)
            


real = ytest_real.tolist()
predict = ytest_pred.tolist()
print(accuracy_score(predict, real))

total_params = sum(p.numel() for p in net.parameters())
print(f'Number of total parameters in the model: {total_params}')

trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Number of trainable parameters in the model: {trainable_params}')

from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device) 
summary(net, (9, 128))

import thop
from thop import profile


input = torch.randn(1, 9, 128).to(device)

# 估计模型的FLOPs
flops, params = profile(net, inputs=(input,))
print(f"FLOPs: {flops / 1e6} M FLOPs")  #百万次浮点运算

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

# plt.plot(smooth_curve(loss_list1), 'bo', label='Smoothed training loss')
# plt.plot(smooth_curve(loss_list2), 'b', label='Smoothed validation loss')

plt.figure(figsize=(10,7))
plt.plot(smooth_curve(loss_list1), label='1.5m', linestyle='-.')
plt.plot(smooth_curve(loss_list2), label='1.8m', linestyle='-.', color='red')
plt.plot(smooth_curve(loss_list3), label='hold randomly')
plt.plot(smooth_curve(loss_list4), label='hold horizontally')
plt.plot(smooth_curve(loss_list5), label='hold vertically')
plt.xlabel('Epochs', fontweight='bold', fontsize=16)
plt.ylabel('Test Loss', fontweight='bold', fontsize=16)
plt.title('Test Loss with Different Height of Users and Different Grip Gesture', fontsize=16)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()                                             