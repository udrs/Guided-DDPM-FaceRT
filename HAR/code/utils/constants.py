## FILE DICTIONARIES
# LOCAL
# TRAINFOLD = "../data/train/Inertial Signals/"
# TESTFOLD = "../data/test/Inertial Signals/"
# TRAINFEATURE = ["../data/train/X_train.txt", "../data/train/y_train.txt"]
# TESTFEATURE = ["../data/test/X_test.txt", "../data/test/y_test.txt"]
# MADETEST = "../data/made_test/test_loader"
# OUR_TESTFOLD = "../data/data_test/"
# RAW_FOLD = "../data/HAPT Data Set/RawData/"

TRAINFOLD = "C:/Users/97233/Desktop/data3/Exp/20210627_101924"
TESTFOLD = "C:/Users/97233/Desktop/data3/Exp/20210627_102255"

'''
TRAINFEATURE = ["D:/myHAR/myHAR/data/train/X_train", "D:/myHAR/myHAR/data/train/y_train.txt"]
TESTFEATURE = ["D:/myHAR/myHAR/data/test/X_test.txt", "D:/myHAR/myHAR/data/test/y_test.txt"]
MADETEST = "D:/myHAR/myHAR/data/made_test/test_loader"
OUR_TESTFOLD = "D:/myHAR/myHAR/data/data_test/"
RAW_FOLD = "D:/myHAR/myHAR/data/HAPT Data Set/RawData/"
'''

# # SERVER
# TRAINFOLD = "/data/419yangnan/datasets/har_data/data/train/Inertial Signals/"
# TESTFOLD = "/data/419yangnan/datasets/har_data/data/test/Inertial Signals/"
# TRAINFEATURE = ["/data/419yangnan/datasets/har_data/data/train/X_train.txt", "/data/419yangnan/datasets/har_data/data/train/y_train.txt"]
# TESTFEATURE = ["/data/419yangnan/datasets/har_data/data/test/X_test.txt", "/data/419yangnan/datasets/har_data/data/test/y_test.txt"]

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
#OUR_TEST = ["/1.txt", "/2.txt", "/3.txt"]
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
        # 7:[2,3],
        # 8:[3,2],
        # 9:[4,5],
        # 10:[5,4],
        # 11:[4,6],
        # 12:[6,4],
        # 13:[2,5],
        # 14:[5,2],
        # 15:[3,5],
        # 16:[5,3]

CLASS_IDV = [1,2,3,5] #individual class
        
SAVEPATH = "save_models/"
# DATE = "2020.8.20/"
DATE = "2021.2.22/"

# DATA SETTING
WINDOW_WIDTH = 128 # MAX_WINDOW_WIDTH = 128
# WINDOW_WIDTH = 128
# WINDOW_WIDTH = 128

STARTPOINT = 0
data_mix = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, "random"] # make test dataset , 0.15, 0.3, 0.45
NUM_CLASSES = 10
num_channels = 9
NUM_FEATURES_USED = 1152 #128x9
SPLIT_RATE = .95
TARGET_NAMES = {60:["Walking",
                   "Walking Upstairs", 
                   "Walking Downstairs", 
                   "Sitting", 
                   "Standing", 
                   "Laying"],
                16:["W to U",
                    "U to W",
                    "W to D",
                    "D to W",
                    "U to D",
                    "D to U",
                    "W to S",
                    "S to W",
                    "Si to St",
                    "St to Si",
                    "S to L",
                    "L to S",
                    "U to S",
                    "S to U",
                    "D to S",
                    "S to D"],
                40:["Walking to Upstairs",
                   "Upstairs to Walking",
                   "Walking to Downstairs",
                   "Downstairs to Walking"],
                6: ["Walking to Upstairs",
                    "Upstairs to Walking",
                    "Walking to Downstairs",
                    "Downstairs to Walking",
                    "Walking to Standing",
                    "Standing to Walking"],
                10: ["Walking to Upstairs",
                    "Upstairs to Walking",
                    "Walking to Downstairs",
                    "Downstairs to Walking",
                    "Walking to Standing",
                    "Standing to Walking",
                    "Walking to Walking",
                    "Upstairs to Upstairs",
                    "Downstairs to Downstairs",
                    "Standing to Standing"],
                4: ["Walking",
                     "Upstairs",
                     "Downstairs",
                     "Standing"]}

# NETWORK CONSTANTS
METHOD = [] #CC "SAE", "CNN_org"
AE1_DIM = 512
AE2_DIM = 8
Conv1d_ft = [WINDOW_WIDTH, int(WINDOW_WIDTH/2)] #conv filter number #128,64
Conv1d_st = [3, 3] #conv stride number
FC_DIM = [150, NUM_CLASSES]


# TRAINING CONSTANTS
EPOCH_NUM = 20
DISPLAY_NUM = 100
BATCH_SIZE = 64
SAVE_EVERY = 20
DEV_EVERY = 419
LOG_EVERY = 419
trans_times = 1
test_size = 1000
DATA_TYPE = 'trans'  #original, features, trans
TRAINCUT_RAND = True  #CC random cut or not
TEST_flag = []  #"OURS"