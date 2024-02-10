import torch
import pandas as pd
from torch.utils.data import Dataset


class EmailDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        features = torch.tensor(row.drop('class').values, dtype=torch.float32)
        label = torch.tensor(row['class'], dtype=torch.long)
        return features, label

    def __len__(self):
        return len(self.dataframe)