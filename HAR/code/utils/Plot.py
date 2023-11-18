import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import time

from sklearn.metrics import *
from utils.constants import *
# from constants import *

WINDOW_WIDTH = 128

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def acc_plot():
    labels = ['32', '64', '128']
    a = np.array([87.393, 89.409, 89.667])
    b = np.array([87.352, 88.839, 90.299])

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, a, width, label='No Validation')
    rects2 = ax.bar(x + width/2, b, width, label='Validation')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acc. %')
    ax.set_xlabel('# of Timesteps')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim((85,91))

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig("plots/acc.jpg")

def heatmap(y_true, y_pred,no_class, name):
    target_names = TARGET_NAMES[no_class]
    if len(set(y_pred)) != no_class and len(set(y_true)) != no_class:
        for i in range(no_class, 0, -1):
            if i not in set(y_pred):
                del target_names[i-1]
    clf_cm = confusion_matrix(y_true, y_pred)
    # if len(set(y_pred)) != no_class:
    #     for i in range(1,no_class):
    #         if i not in set(y_pred):
    #             clf_cm = np.insert(clf_cm, i-1, values=np.zeros(clf_cm.shape[1]), axis=0)
    #             clf_cm = np.insert(clf_cm, i-1, values=np.zeros(clf_cm.shape[0]), axis=1)

    clf_report = classification_report(y_true, y_pred, digits=4, target_names=target_names, output_dict=True)
    clf_report2 = classification_report(y_true, y_pred, digits=4,target_names=target_names)
    print(clf_report2)

    f, ax= plt.subplots(figsize=(6,5))
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :-3].T, 
                annot=True, cmap="Blues_r", fmt=".4f")
    ax.set_title("Classification Report")
    f.tight_layout()
    if os.path.isdir("plots/"+DATE):
        plt.savefig("plots/"+DATE+name+time.strftime('%H_%M_%S')+"_classification_report.jpg")
    else:
        os.mkdir("plots/"+DATE)
        plt.savefig("plots/"+DATE+name+time.strftime('%H_%M_%S')+"_classification_report.jpg")
    f.clf()

    f, ax= plt.subplots(figsize=(5,5))
    #4.23及以前的图横轴True，纵轴Prediction
    sns.heatmap(pd.DataFrame(clf_cm, index=target_names, columns=target_names).T, 
                    annot=True, cmap="BuGn", fmt="d")
    ax.set_title("Confusion Matrix")
    f.tight_layout()
    plt.savefig("plots/"+DATE+name+time.strftime('%H_%M_%S')+"_confusion_matrix.jpg")

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def plot_activity(data, name, seg_len=WINDOW_WIDTH):
    fig, ax = plt.subplots()
    for i in range(num_channels):
        ax.plot(range(0,seg_len), data[i], label=str(1+i))
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(name)
    plt.subplots_adjust(top=0.90)
    plt.legend()
    plt.savefig("D:/myHAR/myHAR/plots/" + DATE + name +".jpg")

def plot_hist(data, name):
    fig, ax = plt.subplots()
    x = ax.hist(data, NUM_CLASSES, rwidth=0.8)
    plt.xticks(x[1]+1, TARGET_NAMES[NUM_CLASSES], rotation=-60, fontsize=8)
    # plt.xticks(range(2, 3+NUM_CLASSES),TARGET_NAMES[NUM_CLASSES],rotation=-60,fontsize=8)
    ax.set_title('Training distribution')
    ax.set_xlabel('Activity')
    ax.set_ylabel('Count')
    plt.subplots_adjust(bottom=0.3, top=0.9)
    for x, y in zip(range(1, 1+NUM_CLASSES), x[0]):
        plt.text(x, y+1, '%d' % y, ha='center')
    plt.savefig("D:/myHAR/myHAR/plots/" + DATE + name + time.strftime('%H_%M_%S') + "_Training_distribution.jpg")

def acc_curve_plt(x_axis,acc1,acc2, name, xlabel):
    fig, ax = plt.subplots()
    if METHOD =="SAE":
        ax.plot(x_axis, acc2, marker='*', color='red')
        ax.set_title('Accuracy using SAE method')
    else:
        ax.plot(x_axis, acc1, marker='o', color='green', label='transition accuracy')
        ax.plot(x_axis, acc2, marker='*', color='red', label='individual class accuracy')
        plt.legend()
        if TRAINCUT_RAND == True:
            postname = "(random cut)"
        else:
            postname = "(50% cut)"
        ax.set_title("Accuracy with mixed data"+ postname)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Test Accuracy %')
    if os.path.isdir("plots/"+DATE):
        plt.savefig("plots/" + DATE + name + time.strftime('%H_%M_%S') + "_accuracy_curve.jpg")
    else:
        os.mkdir("plots/"+DATE)
        plt.savefig("plots/" + DATE + name + time.strftime('%H_%M_%S') + "_accuracy_curve.jpg")

def comp_plt(x_axis,acc1,acc2, acc3,acc4,acc5):
    fig, ax = plt.subplots()
    ax.plot(x_axis, acc1, marker='o', color='red', label='128 points')
    ax.plot(x_axis, acc2, marker='x', color='green', label='64 points')
    ax.plot(x_axis, acc3, marker='+', color='black', label='32 points')
    ax.plot(x_axis, acc4, marker='.', color='blue', label='16 points')
    ax.plot(x_axis, acc5, marker='*', color='yellow', label='8 points')
    plt.legend()
    # ax.set_title('Comparision among three methods')
    ax.set_title('Accuracy against different window size')
    ax.set_xlabel('Test data mixed %')
    ax.set_ylabel('Test Accuracy')
    if os.path.isdir("plots/" + DATE):
        plt.savefig("plots/" + DATE + "Compare methods" + time.strftime('%H_%M_%S') + "_accuracy_curve.jpg")
    else:
        os.mkdir("plots/" + DATE)
        plt.savefig("plots/" + DATE + "Compare methods" + time.strftime('%H_%M_%S') + "_accuracy_curve.jpg")