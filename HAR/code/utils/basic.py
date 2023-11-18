
import glob
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils.Preprocessing import *
from utils.constants import *
from utils.Plot import *


class DealDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.len = x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_loader(status, temp, shuffle=False, validation=False, num_workers=2):

    prepare_data = Preprocessing() # preprocessing is a class
    if status == 'test':
        data_X, data_y = prepare_data.trans(status, temp)

        #print("dataX of data_loader ouput: {}".format(data_X))
        #print("dataY of data_loader ouput: {}".format(data_y))

        return data_X, data_y

    else:
        # Here DATA_TYPE = trans;
        if DATA_TYPE == 'original':
            data_X, data_y = prepare_data.original(status)
        elif DATA_TYPE == 'features':
            data_X, data_y = prepare_data.features(status)
        elif DATA_TYPE == 'trans':
            data_X, data_y = prepare_data.trans(status, temp)

    # data_X = Preprocessing.feature_normalize(np.array(data_X)) #CC (-1,9,128)

    # print("data_X:{}".format(data_X.size()))
    # print("data_Y:{}".format(data_y.size()))

    # data_X:torch.Size([28194, 9, 128])
    # data_Y:torch.Size([28194])

    data = DealDataset(data_X, data_y)
    size = data.len

    if validation:
        # print("validation is ture")
        train, dev = random_split(data, [int(size*SPLIT_RATE), size-int(size*SPLIT_RATE)])
        train, dev = DealDataset(train[:][0],train[:][1]), DealDataset(dev[:][0],dev[:][1])
        train_loader = DataLoader(dataset=train,
                        batch_size=BATCH_SIZE,
                        shuffle=shuffle,
                        num_workers=num_workers)
        dev_loader = DataLoader(dataset=dev,
                        batch_size=BATCH_SIZE,
                        shuffle=shuffle,
                        num_workers=num_workers)
        return train_loader, dev_loader
    else:
        # print("validation is false")
        loader = DataLoader(dataset=data,
                        batch_size=BATCH_SIZE,
                        shuffle=shuffle,
                        num_workers=num_workers)
        return loader

def checkpoint(net, save_path, acc, loss, iterations):
    snapshot_prefix = os.path.join(save_path, 'snapshot_' + net._class_name())
    snapshot_path = snapshot_prefix + '_acc_{:.2f}_loss_{:.4f}_iter_{}_model.pt'.format(acc, loss.item(), iterations)
    torch.save(net, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)
            
def train(optimizer, criterion, net, validation, device, epoches, amount, save_path=SAVEPATH):
    iterations = 0
    start = time.time()

    best_dev_acc = -1; best_snapshot_path = ''
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.1f}%,{:>7.4f},{:8.4f},{:12.4f},{:12.4f}'.split(','))
    log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.1f}%,{:>7.4f},{},{:12.4f},{}'.split(','))
    
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    print(header)

    if validation:
        train_loader, dev_loader = data_loader("train", amount, True, validation)
    else:
        train_loader = data_loader("train", amount, True, validation)

    # plot_hist(train_loader.dataset.y_data, "ConV1D_" + str(NUM_CLASSES) + "_")  # CC

    dev_loss_list = []
    for epoch in range(epoches):  # loop over the dataset multiple times
        correct, total = 0, 0
        for i, data in enumerate(train_loader, 0):
            iterations += 1
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            # print("inputs size:{}".format(inputs.size())) # inputs: 64,9,128

            # inputs = inputs.view(-1,1152) # SAE coding need

            outputs = net(inputs.float())  # torch.Size([32, 6])
            labels = labels.view(-1) - 1  # torch.Size([32, 1])

            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute accuracy
            acc = correct / total * 100

            # checkpoint model periodically
            if iterations % SAVE_EVERY == 0:
                checkpoint(net, save_path, acc, loss, iterations)

            # validation model periodically
            if validation and iterations % DEV_EVERY == 0:
                # calculate accuracy on validation set
                dev_correct, dev_total = 0, 0
                avg_dev_loss = 0
                with torch.no_grad():
                    # for dev_batch_idx, dev_batch in enumerate(dev_loader, 0):
                    for dev_batch_idx, dev_batch in enumerate(dev_loader):
                        signals, labels = dev_batch
                        signals = signals.to(device)
                        labels = labels.to(device)
                        labels = labels.view(-1) - 1

                        # print("signals:{}".format(signals.size()))
                        # signals:torch.Size([64, 9, 128])
                        predicts = net(signals.float())
                        dev_loss = criterion(predicts, labels)
                        avg_dev_loss += dev_loss.item()
                        dev_correct += (torch.max(predicts, 1)[1].view(-1) == labels).sum().item()
                        dev_total += labels.size(0)
                dev_acc = 100. * dev_correct / dev_total
                dev_loss_list.append(avg_dev_loss/dev_total)

                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+i, len(train_loader),
                    100. * (1+i) / len(train_loader), loss.item(), dev_loss.item(), acc, dev_acc))

                # update best valiation set accuracy
                if dev_acc > best_dev_acc:

                    # found a model with better validation set accuracy

                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(save_path, 'best_snapshot_' + net._class_name())
                    best_snapshot_path = snapshot_prefix + '_'+str(WINDOW_WIDTH)+'_devacc_{:.2f}_devloss_{:.4f}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                    # save model, delete previous 'best_snapshot' files
                    torch.save(net, best_snapshot_path)
                    # for f in glob.glob(snapshot_prefix + '*'):
                    #     print('no delete')
                        # if f != best_snapshot_path:
                        #     os.remove(f)

            elif iterations % LOG_EVERY == 0:
                # print progress message
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+i, len(train_loader),
                    100. * (1+i) / len(train_loader), loss.item(), ' '*8, acc, ' '*12))

    print('Finished Training')
    return net, best_snapshot_path, len(train_loader.dataset), dev_loss_list, train_loader


def test(test_loader, net, fp, validation, device):
    correct = 0
    total = 0
    
    # test_loader = data_loader("test", 0, 0)
    # test_loader = data_loader("test", 0, 0)
    
    y_pred = []
    y_true = []
    y_pred_prob = []

    if validation:
        net = torch.load(fp)

    with torch.no_grad():
        #for data in enumerate(test_loader):
        for step, (signals, labels) in enumerate(test_loader):
            # print(len(data))
            # signals, labels = data
            
            # print('signal:{}'.format(signals))
            # print('labels:{}'.format(labels))

            signals = signals.to(device)
            labels = labels.to(device)
            for i in labels.view(-1):
                y_true.append(i.view(-1).tolist())
            
            outputs = net(signals.float())
            predicted = outputs.data.argmax(dim=1)
            for i in predicted.view(-1) + 1:
                y_pred.append(i.view(-1).tolist())
            
            labels = labels.view(-1) - 1
            correct += (predicted.view(-1) == labels).sum().item()
            total += labels.size(0)

            predicted_prob = outputs.data.softmax(dim=1).tolist()
            for i in range(len(predicted_prob)):
                y_pred_prob.append(predicted_prob[i][predicted[i]])


    acc = 100 * correct / total
    # print(total,correct)
    # print('Accuracy: %.2f %%' % acc)

    return acc, np.ravel(y_true), np.ravel(y_pred), y_pred_prob

def test_final(y_true, y_pred, y_pred_prob):
    y_true_org = []
    y_pred_org = []

    for j in range(len(y_pred)):
        y_true_org.append(PAIR[y_true[j]][0])
        y_pred_org.append(PAIR[y_pred[j]])

    y_pred_select = list(np.copy([y_pred_org[0][0]]))
    for i in range(len(y_pred_org)-1):
        if y_pred_prob[i] >= y_pred_prob[i+1]:
            y_pred_select.append(y_pred_org[i][1])
        else:
            y_pred_select.append(y_pred_org[i + 1][0])

    correct = [y_true_org[i] == y_pred_select[i] for i in range(0,len(y_pred_select))]
    acc = 100 * sum(correct)/len(y_true_org)

    return acc, y_true_org, y_pred_select

def save_testset(): # save test data as npz document.
    for i in range(len(data_mix)):
        # temp = the second parameter of data_loader
        if data_mix[i] != "random":
            data_X, data_y = data_loader("test",int(WINDOW_WIDTH*data_mix[i]), 0, 0)
        else:
            data_X, data_y = data_loader("test", "random", 0, 0)

        #print("dataX: {}".format(data_X.shape))
        #print("dataY: {}".format(data_y.shape))

        print("save test set finished")
        np.savez(MADETEST+"_"+str(data_mix[i])+".npz", data_X, data_y)

def read_testset(filename):
    y_org = []
    file = np.load(filename)
    data_X, data_y = file['arr_0'], file['arr_1']
    if METHOD == "SAE" or METHOD == "CNN_org":
        if METHOD == "SAE":
            data_X = data_X.reshape(len(data_X), -1)  # CC:for SAE input(,9,128)->(,-1)
        for j in range(len(data_y)):
            y_org.append(PAIR[data_y[j]][0])
            data = DealDataset(data_X, y_org)
    else:
        data = DealDataset(data_X, data_y)
    # loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=0, num_workers=2)
    loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=0, num_workers=0)
    return loader

def read_our_test(file_path):
    for i in range(len(OUR_TEST)):
        filename = file_path + OUR_TEST[i]
        if i == 0:
            data = pd.read_csv(filename, delimiter=",", dtype=np.float64,usecols=[0,1,2])
        else:
            data = np.concatenate((data, pd.read_csv(filename, delimiter=",", dtype=np.float64,usecols=[0,1,2])),axis = 1) #(-1,9)
    test_set = []
    num = int((len(data)-WINDOW_WIDTH)/(WINDOW_WIDTH/2))
    for i in range(num):
        start_pt = int(i*WINDOW_WIDTH/2)
        test_set.append(data[start_pt:start_pt+WINDOW_WIDTH,:])#(21,128,9)
    test_set = np.array(test_set).swapaxes(1,2)#(21,9,128)
    test_set[:, [0, 2], :] = test_set[:, [2, 0], :]
    test_set[:, [3, 5], :] = test_set[:, [5, 3], :]
    test_set[:, [6, 8], :] = test_set[:, [8, 6], :]
    # plot_activity(test_set[0], "ours1")
    test_set = DealDataset(test_set/10, np.ones(len(test_set)))
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=0, num_workers=2)
    return test_loader

def read_raw_data(file_path):
    total_acc = pd.read_csv(file_path + "acc_exp01_user01.txt", sep="\s+",header=None)
    body_acc = total_acc-9.80665
    gyro = pd.read_csv(file_path + "gyro_exp01_user01.txt", sep="\s+",header=None)
    data = np.concatenate((body_acc,gyro,total_acc),axis = 1) #(-1,9)

    data = data[:WINDOW_WIDTH*int(len(data)/WINDOW_WIDTH),:].reshape(-1,WINDOW_WIDTH,9) #(11,128,9)
    data = data.swapaxes(1,2)
    # data = Preprocessing.feature_normalize(np.array(data)  # CC (-1,9,128)
    plot_activity(data[0], "raw")
    data = DealDataset(data, np.ones(len(data)))
    test_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=0, num_workers=2)
    return test_loader