import torch
from utils.constants import *
from utils.NeuralNetwork import *
from utils.basic import *
from utils.Plot import *

def run(times, validation, gpu, window_width, plot, epoch):
    acc_col = []

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device('cuda:{}'.format(gpu))
        print("Using GPU for training")
    else:
        device = torch.device('cpu')

    for i in range(times):
        print('\n----------------------------- EXPERIMENT %d -----------------------------' % (i+1))
        
        net = DNN(NUM_FEATURES_USED).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()
        net, fp, a, b, train_loader = train(optimizer, criterion, net, validation, device, epoch, 1)

        data_x, data_y = data_loader("test", 0, 0)
        np.savez(MADETEST+"_DNN" +".npz", data_x, data_y)
        filename = MADETEST+"_DNN" +".npz"
        test_loader = read_testset(filename)

        #test_loader = data_loader("test", 0, 0)
        
        acc, y_true, y_pred, y_pred_prob = test(test_loader, net, fp, validation, device)
        acc_col.append(acc)

        if i == times - 1 and plot:
            heatmap(y_true, y_pred, "DNN"+str(NUM_FEATURES_USED))
    
    if times > 1:
        print(np.round(acc_col,2))
        print("Average accuracy of %d experiments is: %.3f %%" % (times, np.mean(acc_col)))
    else:
        print("Accuracy is: %.3f %%" % np.mean(acc_col))

    print("Accuracy is: %.3f %%" % acc)    

    # for j in range(len(data_mix)):
    #     filename = MADETEST + "_" + str(data_mix[j]) + ".npz"
    #     test_loader = read_testset(filename)
    #     acc, y_true, y_pred, y_pred_prob = test(test_loader, net, fp, validation, device)
    #     acc_col[j,i] = acc
    #     np.savez("result.npz", y_true, y_pred, y_pred_prob)

    #     if i == times - 1 and plot:
    #         heatmap(y_true, y_pred, len(CLASS_IDV), "SAE"+str(NUM_FEATURES_USED)+ "_mix_" + str(data_mix[j]) + "_")

    # print("Accuracy of different mixed percent is: ")
    # print(acc_col[:, i])
    # acc_col = np.mean(acc_col, axis=-1)
    # print("Average accuracy of %d experiments is:" % times)
    # print(acc_col)

    acc_mtx_trans = np.zeros(shape=(trans_times, len(data_mix), times))
    acc_mtx_org = np.zeros(shape=(trans_times, len(data_mix), times))
    amount = 0

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

        print("done!")

        print(acc_trans)

        acc_mtx_trans[amount, j, i] = acc_trans
        acc_mtx_org[amount, j, i] = acc_org

        if i == times - 1 and plot:
            heatmap(y_true_trans, y_pred_trans, NUM_CLASSES, "ConV1D_"+str(NUM_CLASSES)+"_mix_"+str(data_mix[j])+"_")
            if data_mix[j] != 0:
                heatmap(y_true_org, y_pred_select, len(CLASS_IDV), "ConV1D_org_" + str(len(CLASS_IDV))+ "_mix_"+str(data_mix[j])+"_")

    print("Transition and individual accuracy of different mixed percent is: ")
    print(acc_mtx_trans[amount, :, i], acc_mtx_org[amount, :, i])