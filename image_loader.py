import torch
import random
import os
import pickle
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
        label = torch.as_tensor(label)
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
        p_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i,batch in p_bar:
            features, labels = batch
            prediction = model(features)
            loss = criteria(prediction, labels)
            loss.backward()            
            optimiser.step() # optimisation step
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
            p_bar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {round(torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels), 2)}, Losses = {round(loss.item(), 2)}")

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        
        # define layers        
        output_features = self.resnet50.fc.out_features
        self.linear = torch.nn.Linear(output_features, 13)

        self.main = torch.nn.Sequential(self.resnet50, self.linear)

    def forward(self, X):
        return self.main(X)

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

    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(dataset.decoder, open('image_decoder.pkl', 'wb'))

