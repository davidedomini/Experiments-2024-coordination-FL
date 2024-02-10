import sys

import pandas

sys.path.append('../common')

import torch
import glob
import copy
import numpy as np
import pandas as pd
from MLP import MLP
import seaborn as sns
from torch import nn, optim
import matplotlib.pyplot as plt
from LocalUpdate import LocalUpdate
from EmailDataset import EmailDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


if __name__ == '__main__':
    # Loading data
    dataframes = []
    files = glob.glob('../data/federated_train/*.csv')
    for file in files:
        d = pd.read_csv(file)
        dataframes.append(d)

    dataframe_test = pandas.read_csv('../data/federated_test/spambase_test.csv')

    # Training
    frac = 0.5  # Fraction of devices used in aggregation
    global_model = MLP(57, 256, 2)
    global_model.train()
    global_weights = global_model.state_dict()
    num_users = len(dataframes)

    train_loss, mean_val_accuracy, mean_val_loss = [], [], []
    val_loss = []
    val_acc_list, net_list = [], []
    cv_loss, cv_accuracy = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    global_epochs = 10

    for epoch in range(global_epochs):
        local_weights, local_losses, local_val_acc, local_val_loss = [], [], [], []
        print(f'\n | Global training round: {epoch + 1} | \n')
        global_model.train()
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(dataset=EmailDataset(dataframes[idx]))
            w, loss, val_acc, val_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                                    global_round=epoch)
            local_val_acc.append(val_acc)
            local_val_loss.append(val_loss)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        mean_val_accuracy.append(sum(local_val_acc) / len(local_val_acc))
        mean_val_loss.append(sum(local_val_loss) / len(local_val_loss))

    x = range(11)
    y = [0] + mean_val_accuracy
    sns.lineplot(x=x, y=y)
    plt.savefig('avg_validation_accuracy.png')
















