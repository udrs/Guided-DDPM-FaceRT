import torch

from utils.constants import *
from utils.NeuralNetwork import *
from utils.Plot import *
from utils.basic import *
from utils.constants import *

# from basic import train

torch.backends.cudnn.benchmark = True

def run(times, validation, gpu, window_width, plot, epoch):
    # trans_times = 1
    acc_mtx_trans = np.zeros(shape=(trans_times, len(data_mix), times))
    acc_mtx_org = np.zeros(shape=(trans_times, len(data_mix), times))
    training_size = []

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device('cuda:{}'.format(gpu))
        print("Using GPU for training")
    else:
        device = torch.device('cpu')

    # file = np.load("D:/myHAR/myHAR/code/11result.npz")
    # y_pred_trans, y_pred_prob, y_pred_select = file['arr_0'], file['arr_1'], file['arr_2']

    # heatmap(y_true_trans, y_pred_trans, NUM_CLASSES, "ConV1D_" + str(NUM_CLASSES) + "_mix_" + str(data_mix[0]) + "_")
    # heatmap(y_true_org, y_pred_select, len(CLASS_IDV),"ConV1D_org_" + str(len(CLASS_IDV)) + "_mix_" + str(data_mix[0]) + "_")

    # comp_plt([int(i * 100) for i in data_mix[:-1]], [85.68, 83.78, 91.86, 95.52, 96, 96.3,  96.54, 96.68, 97.12, 97.62], [95.04, 95.56, 94.26, 94.46, 92.54, 90.32, 88.44, 83.16, 77.58, 73.8],[87.06, 86.58, 86.12, 85.1,  81.2,  80.14, 76.12, 71.22, 66.38, 63.72],[65.32, 66.72, 73.3,  75.94],[48.52, 55.1,  53.08, 58.46])
    # acc_curve_plt([int(i * 100) for i in data_mix[:-1]], [85.76, 82.96, 88.16, 90], [85.76, 93.94, 95.86, 96.44], "ConV1D_" + str(NUM_CLASSES) + "_", 'Mixed test data %')

    save_testset()

    for amount in range(trans_times):
        for i in range(times):
            print('\n-----------------------------Training size %d ---EXPERIMENT %d -----------------------------'% (amount+1, i+1))
            fc1_input_dim = Conv1d_ft[1] * (Conv1d_ft[1] - Conv1d_st[0] + 1) #CC
            net = Conv1DNet(fc1_input_dim).to(device)
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            net, fp, training_size_temp, dev_loss_list,train_loader = train(optimizer, 
                criterion, net, validation, device, epoch, amount+1)

            if TEST_flag == "OURS":
                f = open(str(WINDOW_WIDTH)+"_result.txt", 'a')
                f.write(time.strftime('%H_%M_%S'))
                for k in range(5):
                    FOLDER = OUR_TESTFOLD + str(k+1)
                    test_loader = read_our_test(FOLDER)
                    _, y_true_trans, y_pred_trans, y_pred_prob = test(test_loader, net, fp, validation, device)
                    _, _, y_pred_select = test_final(y_true_trans, y_pred_trans, y_pred_prob)
                    # np.savez(str(k+1)+str(i+1)+"result.npz",  y_pred_trans, y_pred_prob, y_pred_select)
                    f.write("\n--------------EXPERIMENT %d---FOLDER %d--------------\n" % (i+1,k+1))
                    f.write(str(y_pred_select))
                    f.write("\n")
                f.write("\n\n")
                f.close()

            else:
                for j in range(len(data_mix)):
                    # read npz files
                    filename = MADETEST + "_" + str(data_mix[j]) + ".npz"
                    test_loader = read_testset(filename)
                    acc_trans, y_true_trans, y_pred_trans, y_pred_prob = test(test_loader, net, fp, validation, device)
                    
                    if data_mix[j] == 0:
                        acc_org, y_true_org, y_pred_select = acc_trans, y_true_trans, y_pred_trans
                    else:
                        acc_org, y_true_org, y_pred_select = test_final(y_true_trans, y_pred_trans, y_pred_prob)
                    np.savez("result.npz", y_true_trans, y_pred_trans, y_pred_prob,y_true_org, y_pred_select)

                    # print("acc_trans:{}".format(acc_trans))
                    acc_mtx_trans[amount, j, i] = acc_trans
                    acc_mtx_org[amount, j, i] = acc_org

                    if i == times - 1 and plot:
                        heatmap(y_true_trans, y_pred_trans, NUM_CLASSES, "ConV1D_"+str(NUM_CLASSES)+"_mix_"+str(data_mix[j])+"_")
                        if data_mix[j] != 0:
                            heatmap(y_true_org, y_pred_select, len(CLASS_IDV), "ConV1D_org_" + str(len(CLASS_IDV))+ "_mix_"+str(data_mix[j])+"_")

                print("Transition and individual accuracy of different mixed percent is: ")
                print(acc_mtx_trans[amount, :, i], acc_mtx_org[amount, :, i])
        # training_size.append(training_size_temp)
        test_acc_trans = np.round(np.mean(acc_mtx_trans, axis=-1), 2)
        test_acc_org = np.round(np.mean(acc_mtx_org, axis=-1), 2)
        print("Average accuracy of %d experiments for transition and original is: " % times)
        print(test_acc_trans, test_acc_org)
        print(dev_loss_list)

        # acc_curve_plt(training_size,test_acc_trans[amount],test_acc_org[amount], "ConV1D_"+str(NUM_CLASSES)+"_", 'Training Size')
        if METHOD != "SAE" or METHOD != "CNN_org":
            acc_curve_plt([int(i*100) for i in data_mix[:-1]], test_acc_trans[amount][:-1], test_acc_org[amount][:-1], "ConV1D_" + str(NUM_CLASSES) +str(TRAINCUT_RAND)+ "_", 'Test data mixed %')
