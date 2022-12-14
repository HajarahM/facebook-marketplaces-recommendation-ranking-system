import torch
import random
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.transforms import PILToTensor
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def repeat_channel(x):
            return x.repeat(3,1,1)

class ProductsDataset(Dataset):
    def __init__(self, transform: transforms = None, max_length: int=50):

    # Get cleaned product list and extract category labels
        if not os.path.exists('clean_images'):
            raise RuntimeError('Images Dataset not found')
        else:
            self.products = pd.read_csv('cleaned_products.csv', lineterminator='\n')

        self.descriptions = self.products['product_description'].to_list()
        self.labels = self.products['main_category'].to_list()
        self.max_length = max_length
        # Get the Images
        self.files = self.products['image_id']
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x,y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x,y) in enumerate(set(self.labels))}
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label).float()
        image = Image.open('clean_images/' + self.files[index] + '.jpg')
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)
        return image, label

        # sentence = self.descriptions[index]
        # encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        # encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        # with torch.no_grad():
        #     description = self.model(**encoded).last_hidden_state.swapaxes(1,2)

        # description = description.squeeze

        # return image, description, label

    def __len__(self):
        return len(self.files)

def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = random_split(dataset, [train_split, len(dataset) - train_split])
    return train_dataset, validation_dataset

def train(model, epochs=20):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter()
    criteria = torch.nn.CrossEntropyLoss()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in data_loader:
            features, labels = batch
            prediction = model(features)
            loss = criteria(prediction, labels)
            loss.backward()
            print(loss.item())
            optimiser.step() # optimisation step
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
                
        # define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),
            torch.nn.ReLU(), #relu activate
            torch.nn.Conv2d(10, 20, kernel_size=3),
            torch.nn.ReLU(), 
            torch.nn.Flatten(),
            torch.nn.Linear(968000, 16),
            torch.nn.ReLU(), 
            torch.nn.Linear(16, 13),
            torch.nn.Softmax()
        )
        

    def forward(self, X):
        return self.layers(X)

if __name__ =='__main__':
    dataset = ProductsDataset()
    # print(dataset[0][0])
    # print(dataset.decoder[int(dataset[0][1])])
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=True)
    for batch, (data, labels) in enumerate(data_loader):
        print(data)
        print(labels)
        print(data.size())
        if batch==0:
            break
    
    print(dataset[0])
    print(len(dataset))

    model = CNN()
    train(model)    

