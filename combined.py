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
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
from datetime import datetime
from pathlib import Path, PosixPath
from image_loader import ProductsDataset
import warnings
warnings.filterwarnings('ignore')

# Image model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Text model
class TextClassifier(torch.nn.Module):
    def __init__(self,
                 ngpu,
                 input_size: int = 768):
        super(TextClassifier, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(torch.nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(192 , 32))

    def forward(self, inp):
        x = self.main(inp)
        return x

#Combine Models

class CombinedModel(torch.nn.Module):
    def __init__(self, ngpu, input_size = 768, num_classes: int=2):
        super(CombinedModel, self).__init__()
        self.ngpu = ngpu
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # define layers        
        output_features = self.resnet50.fc.out_features
        self.image_classifier = torch.nn.Sequential(self.resnet50, torch.nn.Linear(output_features, 128)).to(device)
        self.text_classifier = TextClassifier(ngpu=ngpu, input_size=input_size)
        self.main = torch.nn.Sequential(torch.nn.Linear(160, 32))
        
    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = random_split(dataset, [train_split, len(dataset) - train_split])
    return train_dataset, validation_dataset

  
def train(model, epochs=5):

    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    writer = SummaryWriter()
    criteria = torch.nn.CrossEntropyLoss()
    epochs = 2

    for epoch in range(epochs):
        hist_acc = []
        hist_loss = []
        p_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (image_features, text_features, labels) in p_bar:
            image_features = image_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            prediction = model(image_features, text_features)
            loss = criteria(prediction, labels)
            loss.backward()
            hist_acc.append(torch.mean((torch.argmax(prediction, dim=1) == labels).float()).item())
            hist_loss.append(loss.item())           
            optimiser.step()                       
            writer.add_scalar('loss', loss.item())
            Losses = round(loss.item(), 2)
            p_bar.set_description(f'Epoch {epoch + 1}/{epochs} Loss: {loss.item():.4f} Acc = {round(torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels), 2)} Total_acc = {round(np.mean(hist_acc), 2)}')
        
    torch.save(model.state_dict(),'final_models/combined_model.pt')
    #save pickle file of decoder dictionary
    pickle.dump(dataset.decoder, open('combined_decoder.pkl', 'wb'))

#load saved model
def load_model():
    checkpoint = torch.load('./final_models/image_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    train.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train.epoch = checkpoint['epoch']
    train.loss = checkpoint['loss']

dataset = ProductsDataset()
model = CombinedModel(ngpu=1, input_size=768, num_classes=dataset.num_classes)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
model.to(device) 

if __name__ =='__main__':
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    print(dataset[0])
    print(len(dataset))

    #run model
    #load_model()
    train(model)




