import sys
sys.path.append('../common')

import torch
import glob
import copy
import utils
import numpy as np
import pandas as pd
import seaborn as sns
from torch import nn, optim
from CNNMnist import CNNMnist
import matplotlib.pyplot as plt
from LocalUpdate import LocalUpdate
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


def sample_dataset(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        dict_users[i] = {int(n) for n in dict_users[i]}
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args['gpu'] else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss

if __name__ == '__main__':
    args = {
        'frac': 0.5,
        'num_users': 10,
        'epochs': 5,
        'num_channels': 1,
        'num_classes': 10,
        'lr': 0.01,
        'wd': 1e-4,
        'local_bs': 10,
        'local_epochs': 2,
        'gpu': None,
        'verbose': 1
    }

    device = 'cpu'

    train_dataset, test_dataset = utils.get_train_test_dataset()
    train_set, val_set = random_split(train_dataset, [50000, 10000])

    user_groups = sample_dataset(train_dataset, 10)

    global_model = CNNMnist()

    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    train_loss, val_accuracy = [], []
    val_loss = []
    val_acc_list, net_list = [], []
    cv_loss, cv_accuracy = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in range(args['epochs']):
        local_weights, local_losses = [], []
        print(f'\n | Global training round: {epoch + 1} | \n')
        global_model.train()
        m = max(int(args['frac'] * args['num_users']), 1)
        idxs_users = np.random.choice(range(args['num_users']), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args['num_users']):
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        val_loss.append(sum(list_loss) / len(list_loss))
        val_accuracy.append(sum(list_acc) / len(list_acc))

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * val_accuracy[-1]))

    df = pd.DataFrame(columns=['Average train loss', 'Average validation loss', 'Average validation accuracy'])
    df['Average train loss'] = train_loss
    df['Average validation loss'] = val_loss
    df['Average validation accuracy'] = val_accuracy
    df.to_csv('../results/results_federated.csv', index=False)

    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f' \n Results after {args["epochs"]} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * val_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))