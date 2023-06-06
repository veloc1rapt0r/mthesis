# Implementation: Oleh Bakumenko, Univerity of Duisburg-Essen

import torch, torchvision, torch.nn as nn
import PIL
import numpy as np
import sys
import os
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creates a pandas dataframe if file exist
# Input:    String -  Path to the .csv log file
# Output:   pandas.DataFrame
def create_df_if_exist(path):
    if exists(path):
        return pd.read_csv(path, sep=',')
    else:
        print('No file exist', path)
        return None

# Creates 4 dataframes, with validate, train, and runtime logs.
# Input:    String -  Path to the .csv log file
# Output:   Tuple of 4 pandas.DataFrame's
def create_dfs_filename(path):
    path_val = path + '_val.csv'
    path_train =  path + '_train.csv'
    path_test = path + '_test.csv'
    path_runtime = path + '_runtime.csv'
    return create_df_if_exist(path_val), create_df_if_exist(path_train), create_df_if_exist(path_test), create_df_if_exist(path_runtime)

def plot_confusion_matrix(path, print_per_class_acc = True, title_above = None):
    if exists(path):
            conf_matr = torch.load(path)
    else:
        print('No file exist', path)
        return None
    per_class_accuracy = 100*conf_matr.diag()/conf_matr.sum(1)
    mean = torch.mean(per_class_accuracy)
    per_class_accuracy_np = np.array2string(np.round(per_class_accuracy.detach().numpy(),2), separator=' ; ')
    mean_np = np.array2string(np.round(mean.detach().numpy(),2))
    class_names = ['0', '  1', ' 2']
    df_cm = pd.DataFrame(conf_matr, class_names, class_names)
    plt.figure()
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    if (title_above is not None):
        plt.title(title_above, y = 1.05)
    if print_per_class_acc:
        plt.title(f"Per class  accuracy:  { per_class_accuracy_np}",y = -0.2)
    plt.show()


def plot_per_class_acc(list_of_tensors, list_of_names, list_of_colors = None, figsize = (15, 5), ylim = (0,100), title = None):
    if list_of_colors is None:
        list_of_colors= ['royalblue','green','purple','orange','deepskyblue','firebrick']
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for i, per_class_acc in enumerate(list_of_tensors):
        axs[0].plot(per_class_acc[:-1,0].detach().numpy(), label=list_of_names[i], color=list_of_colors[i])
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('Validation Accuracy (%)')
        axs[0].set_title('Target 0')
        axs[0].legend(loc =4)
        axs[0].grid(True)
        axs[0].set_ylim(ylim)

        axs[1].plot(per_class_acc[:-1,1].detach().numpy(), label= list_of_names[i], color=list_of_colors[i])
        axs[1].set_xlabel('epochs')
        axs[1].legend(loc =4)
        axs[1].set_title('Target 1')
        axs[1].grid(True)
        axs[1].set_ylim(ylim)

        axs[2].plot(per_class_acc[:-1,2].detach().numpy(), label=list_of_names[i], color=list_of_colors[i])
        axs[2].set_xlabel('epochs')
        axs[2].legend(loc =4)
        axs[2].set_title('Target 2')
        axs[2].grid(True)
        axs[2].set_ylim(ylim)
        fig.tight_layout(pad=2)
    if title is not None:
        fig.suptitle(title, fontsize = 16, y = 1.02)
    plt.show()


