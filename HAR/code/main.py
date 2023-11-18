import argparse
from utils.basic import save_testset
from utils.constants import *
from CNN import run as CNN_run
from SAE import run as SAE_run
from DNN import run as DNN_run

from utils.constants import WINDOW_WIDTH, EPOCH_NUM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--times', type=int, help='Number of running the experiments',
                        default=1)
    parser.add_argument('-v', '--validation', type=int, help='Whether to implement validation',
                        default=1)
    parser.add_argument('-g', '--gpu', type=int, help='Which GPU to use',
                        default=0)
    parser.add_argument('-w', '--window_width', type=int, help='Window width used for training',
                        default=WINDOW_WIDTH)
    parser.add_argument('-p', '--plot', type=int, help='Window to plot the results',
                        default=0)
    parser.add_argument('-e', '--epoch', type=int, help='Number of epoch', default=20)  #25 for SAE # 20 for DNN
    # parser.add_argument('-e', '--epoch', type=int, help='Number of epoch', default=1)
    args = parser.parse_args()

    CNN_run(**vars(args))
    #DNN_run(**vars(args))
    #SAE_run(**vars(args))