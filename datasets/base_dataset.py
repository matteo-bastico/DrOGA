import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

torch.random.manual_seed(42)


class DatasetBase(Dataset):
    def __init__(self, df, labels='gt'):
        # Load variants from file
        self.labels = df[labels]
        self.samples = df.loc[:, df.columns != labels]
        self.labels = self.labels.to_numpy().astype(int)
        self.samples = self.samples.to_numpy()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx, :].astype(float), self.labels[idx].astype(float)


# For testing purposes
if __name__ == '__main__':
    csv_path = "../data/variants_rankscore.csv"
    dataset = DatasetBase(csv_path)
    print("Dataset loaded correctly.")
