import torch
import pandas as pd
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor
from sklearn.datasets import load_diabetes


class ProductsDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv('cleaned_products.csv')

    def __getitem__(self, index):
        example = self.data.iloc[index]
        features = torch.tensor(example[:4])
        label = example[-1]
        return (features, label)

    def __len__(self):
        return len(self.data)

dataset = ProductsDataset()
print(dataset[10])
print(len(dataset))

class DiabetesDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = load_diabetes(return_X_y=True)

    def __getitem__(self, index):
        return (torch.tensor(self.X[index]), torch.tensor(self.y[index]))

    def __len__(self):
        return len(self.X)

dataset = DiabetesDataset()
print(dataset[10])
print(len(dataset))

# TRANSFORMS
dataset = MNIST(root='./data', download=True, train=True, transform=PILToTensor())

example = dataset[0]
features, label = example

print(features)

# DATALOADER
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch in train_loader:
    print(batch)
    features, labels = batch
    print(features.shape)
    print(labels.shape)
    break