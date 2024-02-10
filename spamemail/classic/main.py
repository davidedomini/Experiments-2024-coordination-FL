import sys
sys.path.append('../common')

import torch
import numpy as np
import pandas as pd
from MLP import MLP
import seaborn as sns
from torch import nn, optim
import matplotlib.pyplot as plt
from EmailDataset import EmailDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

def train_single_epoch(model, criterion, optimizer, train_data_loader):
  training_loss = 0
  batch_loss = []
  for batch_index, (features, labels) in enumerate(train_data_loader):
    model.zero_grad()
    outputs = model(features)
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

    # Loading data
    dataframe = pd.read_csv('../data/spambase.csv')
    data = EmailDataset(dataframe)
    train_set, test_set = random_split(data, [4140, 461])
    train_set, val_set = random_split(train_set, [4000, 140])
    train_data_loader = DataLoader(train_set, batch_size=25, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=25, shuffle=False)
    test_data_loader = DataLoader(test_set, batch_size=25, shuffle=False)

    # Defining the model and others learning components
    model = MLP(57, 256, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training
    epochs = 10
    epoch_losses = []
    last_losses = []
    average_training_losses = []
    average_validation_losses = []
    average_validation_accuracies = [0]
    correct, total = 0, 0

    for epoch in range(epochs):
        print(f'Epoch {epoch}')

        model.train()
        last_loss, avg_loss = train_single_epoch(model, criterion, optimizer, train_data_loader)
        average_validation_losses.append(avg_loss)
        last_losses.append(last_loss)

        # Evaluation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_data_loader):
                val_outputs = model(images)
                vloss = criterion(val_outputs, labels)
                y_pred = torch.max(torch.exp(val_outputs), 1)[1]
                correct += torch.sum(torch.eq(y_pred, labels)).item()
                total += len(labels)
                val_loss += vloss
        val_accuracy = correct / total
        l = val_loss / (i + 1)
        print(f'EVAL EPOCH {epoch} ----- Loss {l}')
        average_validation_losses.append(l)
        average_validation_accuracies.append(val_accuracy)

    x = range(epochs + 1)
    sns.lineplot(x=x, y=average_validation_accuracies)
    plt.savefig('accuracy_plot.png')

    # Testing
    y_pred = []
    y_true = []

    model.eval()
    for images, labels in test_data_loader:
        output = model(images)
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    classes = [str(i) for i in range(2)]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
