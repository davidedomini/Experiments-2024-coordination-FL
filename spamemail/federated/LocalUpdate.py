import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

class LocalUpdate(object):
    def __init__(self, dataset):
      self.device = 'cpu'
      self.trainloader, self.validloader = self.train_val(dataset)
      self.criterion = nn.NLLLoss().to(self.device)
      self.lr = 0.01
      self.wd = 1e-4
      self.epochs = 4


    def train_val(self, dataset):
      #data = SpamDataset(dataframe=dataset)
      train_set, val_set = random_split(dataset, [450, 50])
      train_data_loader = DataLoader(train_set, batch_size=25, shuffle=True)
      val_data_loader = DataLoader(val_set, batch_size=25, shuffle=False)
      return train_data_loader, val_data_loader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (features, labels) in enumerate(self.trainloader):
                features, labels = features.to(self.device), labels.to(self.device)

                model.zero_grad()
                outputs = model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                #self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        val_acc, val_loss = self.validation(model)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), val_acc, val_loss

    def validation(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (features, labels) in enumerate(self.validloader):
            features, labels = features.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(features)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss