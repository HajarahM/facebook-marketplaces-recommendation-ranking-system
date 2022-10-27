import torch
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor

# combine image and product detail dataset to index image to details
products_df = pd.read_csv('cleaned_products.csv', lineterminator='\n')
images_df = pd.read_csv('images.csv', lineterminator='\n')
combined_df = pd.concat([products_df, images_df], axis=2)

class ProductsDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv('cleaned_products.csv', lineterminator='\n')

    def __getitem__(self, index):
        example = self.data.iloc[index]
        features = torch.tensor(example[-2])
        label = example[3]
        return (features, label)

    def __len__(self):
        return len(self.data)

dataset = ProductsDataset()
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
for batch in data_loader:
    print(batch)
    features, labels = batch
    # print(features)
    # print(labels)
    break

print(dataset[0])
print(len(dataset))
