#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy.lib.function_base import sinc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from glob import glob
import os


def load_data(file_path):
    data = pd.read_csv(file_path, sep="\s+")
    return data


# In[2]:


train_data_list = []
exp_data_list = []

moves = ['Test1', 'Turn2', 'TurnRight', 'UpTurnUp', 'UpWalk', 'WalkUp', 'Exp', 'Exp2'] # Exp2 for testing
files = ['AccX_0.txt', 'AccY_0.txt', 'AccZ_0.txt', 'GyroX_0.txt', 'GyroY_0.txt', 'GyroZ_0.txt', 'Label_0.txt']

#moves = ['train', 'test']
for move in moves:
    
    folders = glob(os.path.join('./data',move,'*'))
    folders.sort()
    
    for folder in folders:
        
        txts = glob(os.path.join(folder,'*.txt'))
        txts.sort()
        # body_accX, body_accY, body_accZ, 
        # Body_gyroX, Body_gyroY, Body_gyroZ, 
        # Total_accX, Total_accY, Total_accZ
        Body_acc_list = []
        Gyro_list = []
        Total_acc_list = []
        
        for file in files:
            
            csv_path = os.path.join(folder,file)
            var = pd.read_csv(csv_path, header=None, index_col=None).values
            
            if 'Acc' in file: # 
                G = np.ones((var.shape)) * 9.81
                body_var = var - G
                Total_acc_list.append(var)
                Body_acc_list.append(var)
                
            elif 'Gyro' in file:
                Gyro_list.append(var)
                
            elif 'Label' in file:
                label = var
        
        data_list = Body_acc_list + Gyro_list + Total_acc_list + [label]
        
        lengths = [item.shape[0] for item in data_list]
        length = min(lengths)
        
        data_list = [item[:length,:].T for item in data_list]
        data = np.vstack(data_list) # 10 x 3xxx
        
        if move == 'Exp2':
            exp_data_list.append(data)
        else:
            train_data_list.append(data)


############################################################ read and build orignal train_exp dataset ####################################
print('train size:', len(train_data_list))

print('exp length:', len(exp_data_list))

train1 = train_data_list
exp1 = exp_data_list

#print(train1[0].shape)

def overlap(data_np, window_size=128, overlap=16):
    # print(data_np.shape)

    length = data_np.shape[1]
    start_point = 0
    stop_point = start_point + window_size
    overlap_data = data_np[:, start_point:stop_point]
    overlap_data = overlap_data[np.newaxis, :]
    # print('overlap size:', overlap_data.shape)

    while(stop_point <= length-overlap):
        start_point = start_point + overlap
        stop_point = start_point + window_size
        piece = data_np[:, start_point:stop_point]
        piece = piece[np.newaxis, :]
        overlap_data = np.concatenate((overlap_data, piece),axis=0)
        
    # print('overlap size:', overlap_data.shape)
    return overlap_data

tensor = overlap(train1[4])
print(tensor.shape)
# In[42]:


for i in range(0, len(train1)):
    
    if (i==0):
        train2 = overlap(train1[i])
    else:
        pack = overlap(train1[i])
        train2 = np.concatenate((train2, pack))

print("train2 size:", train2.shape)


# In[54]:


train1[0].shape


# In[55]:


pack.shape


# In[56]:


train2.shape


# In[44]:


list_exp2 = []
for i in range(0, len(exp1)):
    if (i==0):
        exp2 = overlap(exp1[i],128,32)
    else:
        pack = overlap(exp1[i],128,32)
        exp2 = np.concatenate((exp2, pack))

    pack2 = overlap(exp1[i],128,32)
    list_exp2.append(pack2)

print("exp2 size:", exp2.shape)


# In[46]:


def getLabel2(input_label):
    PAIR = {1:[1,2], 
            2:[2,1],
            3:[1,3],
            4:[3,1],
            5:[1,4],
            6:[4,1],
            7:[1,5],
            8:[5,1],
            9:[3,5],
            10:[5,3],
            11:[4,5],
            12:[5,4],
            13:[1,1],
            14:[2,2],
            15:[3,3],
            16:[4,4],
            17:[5,5]}

    middel = []
    length1 = 0
    for label in input_label:
        if (len(middel)==0):
            middel.append(label)
        if (label != middel[0]):
            middel.append(label)
        if (len(middel) == 2):
            break
        length1 = length1 + 1

    if (len(middel)==1):
        middel.append(middel[0])

    label2 = list(PAIR.keys())[list(PAIR.values()).index(middel)]

    rate = length1/128

    rate = (100*rate)//10

    return label2, rate


def CreatLabel(data_np):

    label2_list = []
    rate_list = []
    for i in range(0, data_np.shape[0]):

        label_list = data_np[i,9,:].flatten().tolist()
    
        label2, rate = getLabel2(label_list)
        label2_list.append(label2)

        rate_list.append(rate)
    
    return label2_list, rate_list


train_label, rate_train_list = CreatLabel(train2)
with open("Train_dataY.txt", "w+") as textfile:
    for element in train_label:
        textfile.write(str(element) + "\n")

exp_label, rate_list_exp = CreatLabel(exp2)
with open("Test_Exp_dataY.txt", "w+") as textfile:
    for element in exp_label:
        textfile.write(str(element) + "\n")

print(len(train_label), len(exp_label))


# In[47]:


train_dataX = train2[:,0:9,:]
exp_dataX = exp2[:,0:9,:] 

exp_label_list = []
for i in range(0, len(list_exp2)):
    exp2_label,rate_list_exp2 = CreatLabel(list_exp2[i])
    exp_label_list.append(exp2_label)

exp_dataX_list = []
for i in range(0, len(list_exp2)):
    exp_dataX_list.append(list_exp2[i][:,0:9,:])

print(train_dataX.shape,len(train_label))
print(exp_dataX.shape,len(exp_label))


# In[48]:


##################################### To filter five individual activity labels #######################

train_label_individual = []
exp_label_individual = []

for i in range(0, len(train_label)):
    if (train_label[i]==13):
        label = 1
    elif(train_label[i]==14):
        label = 2
    elif(train_label[i]==15):
        label = 3
    elif(train_label[i]==16):
        label = 4
    elif(train_label[i]==17):
        label = 5
    else:
        label = 0 #### transition activity

    if (i==0):
        train_dataX_individual = train_dataX[0,:,:]
        train_dataX_individual = np.reshape(train_dataX_individual,(1,9,128))
        train_label_individual.append(label)
    
    if(label!=0):
        train_label_individual.append(label)
        train_dataX_individual = np.concatenate((train_dataX_individual, np.reshape(train_dataX[i,:,:],(1,9,128))))

print('individual train dataX shape:{}'.format(train_dataX_individual.shape))
print(len(train_label_individual))

# In[49]:


np.savez("Train_train.npz", train_dataX, train_label)
np.savez("Test_Exp_test.npz", exp_dataX, exp_label)
np.savez("Test_Exp_exp0.npz", exp_dataX_list[0], exp_label_list[0])

print('exp0 size:', len(exp_label_list[0]))

np.savez("Test_Exp_exp1.npz", exp_dataX_list[1], exp_label_list[1])

print('exp1 size:', len(exp_label_list[1]))

np.savez("Test_Exp_exp2.npz", exp_dataX_list[2], exp_label_list[2])

print('exp2 size:', len(exp_label_list[2]))


# In[50]:


exp_dataX_list[1].shape


# In[ ]:




