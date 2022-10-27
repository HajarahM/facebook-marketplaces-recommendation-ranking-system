import torch
import requests
import random
import os
import json
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor

class ProductsDataset(Dataset):
    def __init__(self, labels_level: int = 0, transform: transforms = None, merge: bool = False, download: bool = False):
        super().__init__()
        self.data = pd.read_csv('cleaned_products.csv', lineterminator='\n')

    def __getitem__(self, index):
        example = self.data.iloc[index]
        features = torch.tensor(example[-1])
        label = example[3]
        return (features, label)

    def __len__(self):
        return len(self.data)


if __name__ =='__main__':
    dataset = ProductsDataset(merge=True)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in data_loader:
        print(batch)
        features, labels = batch
        # print(features)
        # print(labels)
        break

    print(dataset[0])
    print(len(dataset))
