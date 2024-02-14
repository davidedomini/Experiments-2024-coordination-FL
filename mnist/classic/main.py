import sys
sys.path.append('../common')

import os
import copy
import torch
import utils
import numpy as np
import pandas as pd
import seaborn as sn
from torch import nn, optim
from CNNMnist import CNNMnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split


def train_single_epoch(model, criterion, optimizer, train_data_loader):
    training_loss = 0
    batch_loss = []
    for batch_index, (images, labels) in enumerate(train_data_loader):
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        batch_loss.append(loss.item())

        if (batch_index % 100) == 99:
          last_loss = training_loss / 100
          print(f'batch {batch_index} ------ Loss {last_loss}')
          training_loss = 0

    return last_loss, sum(batch_loss)/len(batch_loss)


if __name__ == '__main__':

    train_dataset, test_dataset = utils.get_train_test_dataset()
    train_set, val_set = random_split(train_dataset, [50000, 10000])

    bs = 50  # batch size
    lr = 0.001  # learning rate
    wd = 1e-4  # weight decay
    epochs = 5

    train_data_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    model = CNNMnist()
    criterion = nn.NLLLoss()
    epoch_loss = []
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    avg_training_loss = []
    last_losses = []
    avg_val_loss = []
    val_acc = []
    correct = 0
    total = 0

    for e in range(epochs):
        print(f'EPOCH {e}')

        # training
        model.train()
        last_loss, avg_loss = train_single_epoch(model, criterion, optimizer, train_data_loader)
        avg_training_loss.append(avg_loss)
        last_losses.append(last_loss)

        # evaluation
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_data_loader):
                val_outputs = model(images)
                vloss = criterion(val_outputs, labels)
                y_pred = torch.max(torch.exp(val_outputs), 1)[1]
                correct += torch.sum(torch.eq(y_pred, labels)).item()
                total += len(labels)
                val_loss += vloss.item()
        val_accuracy = correct / total
        l = val_loss / (i + 1)
        print(f'EVAL EPOCH {e} ----- Loss {l}')
        avg_val_loss.append(l)
        val_acc.append(val_accuracy)

    df = pd.DataFrame(columns=['Average train loss', 'Average validation loss', 'Average validation accuracy'])
    df['Average train loss'] = avg_training_loss
    df['Average validation loss'] = avg_val_loss
    df['Average validation accuracy'] = val_acc
    df.to_csv('../results/results_classic.csv', index=False)




























