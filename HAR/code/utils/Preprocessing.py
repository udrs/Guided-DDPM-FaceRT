import torch
import random
from utils.Plot import *

class Preprocessing():
    """ A Preprocessing Method Collection """
    def __init__(self):
        TRAINFOLD = "C:/Users/97233/Desktop/data3/Exp/20210627_101924"
        TESTFOLD = "C:/Users/97233/Desktop/data3/Exp/20210627_102255"
        TRAINLIST = [
            "AccX_0.txt",
            "AccY_0.txt",
            "AccZ_0.txt",
            "GyroX_0.txt",
            "GyroY_0.txt",
            "GyroZ_0.txt",
            "MagX_0.txt",
            "MagY_0.txt",
            "MagZ_0.txt"
            ]
        TESTLIST = [

            ]
        # TRAINFEATURE = ["D:/myHAR/myHAR/data/train/X_train.txt", "D:/myHAR/myHAR/data/train/y_train.txt"]
        TRAINFEATURE = ["D:/myHAR/myHAR/data/train/X_train", "D:/myHAR/myHAR/data/train/y_train.txt"]

        TESTFEATURE = ["D:/myHAR/myHAR/data/test/X_test.txt", "D:/myHAR/myHAR/data/test/y_test.txt"]
        self.fd = {"train": TRAINFOLD, "test": TESTFOLD}
        self.fl = {"train": TRAINLIST, "test": TESTLIST}
        self.ft = {"train": TRAINFEATURE, "test": TESTFEATURE}


    def _load_data(self, file_path, data_type='original', flag=0):
        # print("filt_path"+file_path)
        STARTPOINT = 0
        WINDOW_WIDTH = 128
        if data_type == 'original':
            if flag == 1:
                data = pd.read_csv(file_path, sep="\s+").iloc[:, 0]  #所有行，第0列
            else:
                # cut first 128 columns. (取值有问题，只有前128列吗)
                data = pd.read_csv(file_path, sep="\s+").iloc[:,STARTPOINT:STARTPOINT+WINDOW_WIDTH] #0-127列
        elif data_type == 'features':
            data = pd.read_csv(file_path, sep="\s+")

        # the return value "data" is DataFrame
        return data

    def _choose_status(self, status="train"):
        if status == "train":
            return self.fd["train"], self.fl["train"], self.ft["train"]
        else:
            return self.fd["test"], self.fl["test"], self.ft["test"]
    
    def _get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1) - 1
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)   
        size.append(N) 
        return ones.view(*size)

    def feature_normalize(self, dataset): #CC (-1,9,128)
        mu = np.mean(dataset, axis=-1)
        sigma = np.std(dataset, axis=-1)
        mu = mu[:, :, None]
        sigma = sigma[:, :, None]
        return (dataset - mu) / sigma

    def _concatenate(self, pair, dataX, dataY, amount, random):
        WINDOW_WIDTH = 128 #  
        actionA = dataX[np.argwhere(dataY==pair[0])[:,0]]
        actionB = dataX[np.argwhere(dataY==pair[1])[:,0]]
        for n in range(amount):
            temp = np.copy(actionA)
            temp2 = np.copy(actionA)  # CC
            for i in range(len(actionA)):
                if random:
                    cut = np.random.randint(1, WINDOW_WIDTH) # if random, cut randomly
                else:
                    cut =int(WINDOW_WIDTH/2) # if not, cut half
                selection = np.random.randint(0, len(actionB))
                temp[i, :, cut:] = actionB[selection, :, cut:]
                temp2[i, :, :WINDOW_WIDTH - cut] = actionA[i, :, cut:]  # CC 把A后放在A前
                temp2[i, :, WINDOW_WIDTH - cut:] = actionB[selection, :, :cut]  # CC A后+B前
            if n == 0:
                output = temp
                output2 = temp2  # CC
            else:
                output = np.concatenate((output,temp), axis=0)
            output = np.concatenate((output, output2), axis=0) #CC A前_B后,A后_B前
            if pair[0] == pair[1]:
                output = np.concatenate((output, actionA), axis=0) #CC 完美切出每类128
        return len(output), output
    
    def _stack(self, dataX, fl, fd):
        for i in range(1, len(fl)):
            fp = fd + fl[i]
            new = self._load_data(fp)
            if i == 1:
                dataX = np.stack([dataX,new],axis=len(dataX.shape))
                # print('second: ')
                # print(dataX.shape)
            else:
                dataX = np.dstack([dataX,new])
                # print('then: ')
                # print(dataX.shape)
        dataX = np.swapaxes(dataX,1,2)
        # print('finally: ')
        return dataX
    
    def original(self, status):
        fd, fl, ft = self._choose_status(status)
        dataX = self._load_data(fd + fl[0])
        dataX = self._stack(dataX, fl, fd)
        dataY = np.asarray(self._load_data(ft[1]))
        outputX = []
        outputY = []
        for i in CLASS_IDV:
            temp = dataX[np.argwhere(dataY==i)[:,0]]
            if i == CLASS_IDV[0]:
                outputX = temp
                outputY = np.repeat(i, len(temp))
            else:
                outputX = np.concatenate((outputX, temp), axis=0)
                outputY = np.concatenate((outputY, np.repeat(i, len(temp))), axis=0)
        if METHOD == "SAE":
            outputX = outputX.reshape(len(outputX),-1) #CC:for SAE input(,9,128)->(,-1)

        dataX = torch.FloatTensor(outputX)
        dataY = torch.LongTensor(outputY)
        #dataY = self._get_one_hot(dataY, NUM_CLASSES)
        return dataX, dataY  #(7351,9,128), 7351
    
    def features(self, status):
        fd, fl, ft = self._choose_status(status)
        dataX = torch.FloatTensor(np.asarray(self._load_data(ft[0], 'features').iloc[:,0:NUM_FEATURES_USED]))
        dataY = torch.LongTensor(np.asarray(self._load_data(ft[1], 'features')))
        return dataX, dataY

    def statistics(self, status):
        fd, fl, ft = self._choose_status(status)
        dataX = np.asarray(self._load_data(ft[0], 'features').iloc[:,0:NUM_FEATURES_USED])
        dataY = np.asarray(self._load_data(ft[1], 'features'))
        return dataX, dataY


    #CC if test, data_stride = temp; if train, amount = temp
    def trans(self, status, temp): 
        fd, fl, ft = self._choose_status(status)

        # 9 channel stacked
        dataX = self._load_data(fd + fl[0]) # (2946, 128)
        # print('first: ')
        # print(dataX.shape)
        dataX = self._stack(dataX, fl, fd) #  test (2946, 9, 128); train (7351, 9, 128)

        # ft[1] = y_train: label
        dataY = np.asarray(self._load_data(ft[1], flag=1).values) #(2946, )

        #print('dataX shape:', dataX.shape)
        #print('dataY shape:', dataY.shape)

        PAIR = {1:[1,2], 
                2:[2,1],
                3:[1,3],
                4:[3,1],
                5:[1,5],
                6:[5,1],
                7:[1,1],
                8:[2,2],
                9:[3,3],
                10:[5,5]}

        if status == 'test':  # CC make testing dataset
            # test_size = 1000
            # CLASS_IDV = [1,2,3,5] #individual class

            data_stride = temp
            for i in range(test_size):
                if data_stride == 0:
                    y1 = random.sample(CLASS_IDV, 1)
                    idxA = random.choice(np.argwhere(dataY == y1)[:, 0])
                    data = np.copy(dataX[idxA])
                    label = list(filter(lambda k: PAIR[k] == sum([y1, y1], []), PAIR))
                else:
                    if i == 0:
                        y1 = random.sample(CLASS_IDV, 1)
                    idxA = random.choice(np.argwhere(dataY == y1)[:, 0]) #CC
                    y2 = random.sample(CLASS_IDV, 1)
                    while sum([y1,y2],[]) not in PAIR.values():
                        y2 = random.sample(CLASS_IDV, 1)
                    idxB = random.choice(np.argwhere(dataY == y2)[:, 0])
                    if temp == "random":
                        data_stride = np.random.randint(1, WINDOW_WIDTH/2)
                    
                    # test dataset directly use np.concatenate
                    data = np.concatenate((dataX[idxA, :, data_stride:], dataX[idxB, :, :data_stride]), axis=-1)

                    label = list(filter(lambda k: PAIR[k] == sum([y1, y2], []), PAIR))
                    y1 = y2
                    idxA = idxB
                if i == 0:
                    outputX = torch.FloatTensor(data[None,:,:])
                    outputY = torch.LongTensor(label)
                else:
                    outputX = torch.cat([outputX, torch.FloatTensor(data[None,:,:])], 0)
                    outputY = torch.cat([outputY, torch.LongTensor(label)], 0)
                # outputX: [1000, 9, 128]
                # outputY: 1000
                # print('outputX: ', outputX.shape)
                # print('outputY: ', outputY.shape)

        else: # make training dataset
            amount = temp
            NUM_CLASSES = 10
            TRAINCUT_RAND = True
            for i in range(NUM_CLASSES):
                pair = PAIR[i+1]
                num, concatenation = self._concatenate(pair, dataX, dataY, amount, TRAINCUT_RAND)
                #concatenation =concatenation.reshape(len(concatenation),-1) #CC:for SAE input(,9,128)->(,-1)
                if i == 0:
                    outputX = torch.FloatTensor(concatenation)
                    outputY = torch.LongTensor(np.repeat(i+1, num))
                else:
                    outputX = torch.cat([outputX, torch.FloatTensor(concatenation)])
                    outputY = torch.cat([outputY, torch.LongTensor(np.repeat(i+1, num))])
                # plot_activity(concatenation[0], '9 channel of '+TARGET_NAMES[10][i]) # SAE runnint error
                # print('outputX: ', outputX.shape)
                # print('outputY:', outputY.shape)
                # print(outputY)
        return outputX, outputY #(22559,128),22555


if __name__ == "__main__":

    #validation = True
    #train_loader, dev_loader = data_loader("train", amount, True, validation)

    prepare_data = Preprocessing()
    print(prepare_data._load_data("C:/Users/97233/Desktop/code/2.csv"))

    training_data_X, training_data_y = prepare_data.original("train")
    testing_data_X, testing_data_y = prepare_data.original("test")
    print(training_data_X.shape, training_data_y.shape)
    print(testing_data_X.shape, training_data_y.shape)

