import torch
import os
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.tensorboard import SummaryWriter

class TextDataset(torch.utils.data.Dataset):
    '''
    The TextDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    Parameters
    ----------
    root_dir: str
        The directory of the CSV with the products details
    Attributes
    ----------
    labels: set
        Contains the label of each sample
    encoder: dict
        Dictionary to translate the label to a 
        numeric value
    decoder: dict
        Dictionary to translate the numeric value
        to a label
    '''

    def __init__(self,
                 root_dir: str = 'cleaned_products.csv',
                 max_length: int=50):
        
        self.root_dir = root_dir
    # Get cleaned product list, extract category labels and description
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"The file {self.root_dir} does not exist")
        else:
            self.products = pd.read_csv(self.root_dir, lineterminator='\n')

        self.labels = self.products['main_category'].to_list()
        self.descriptions = self.products['product_description'].to_list()

        self.max_length = max_length
        self.num_classes = len(set(self.labels))

        #Encoder/Decoder
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        #Bert tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()

    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)

        description = description.squeeze(0)

        return description, label

    def __len__(self):
        return len(self.labels)

class Classifier(nn.Module, TextDataset):
    def __init__(self, ngpu, input_size: int = 768, num_classes: int = 2):
        super(Classifier, self).__init__()
        num_classes = len(TextDataset())
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(192 , 32),
                                  nn.ReLU(),
                                  nn.Linear(32, num_classes))

    def forward(self, inp):
        x = self.main(inp)
        return x

def train(model, epochs=2):

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    criteria = torch.nn.CrossEntropyLoss()
    batch_idx = 0
    losses = []

    for epoch in range(epochs):
        p_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        hist_accuracy = []
        accuracy = 0

        for i, batch in p_bar:
            data, labels = batch
            data.to(device)
            labels.to(device)
            optimiser.zero_grad()
            prediction = model(data)
            loss = criteria(prediction, labels)
            loss.backward()            
            optimiser.step() # optimisation step            
            writer.add_scalar('Loss', loss.item(), batch_idx)
            accuracy = round(torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels), 2)
            writer.add_scalar('Accuracy', accuracy, batch_idx)
            hist_accuracy.append(accuracy)
            batch_idx += 1  
            losses = round(loss.item(), 2)          
            #losses.append(loss.item())
            p_bar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {accuracy}, Losses = {losses}")

    torch.save(model.state_dict(), 'text_model.pt')
    #save pickle file of decoder dictionary
    pickle.dump(dataset.decoder, open('text_decoder.pkl', 'wb'))
    
ngpu=2
model = Classifier(ngpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
model.to(device)

if __name__ == '__main__':
    dataset = TextDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    for batch, (data, labels) in enumerate(dataloader):
        print(data)
        print(labels)
        if batch == 0:
            break

    #run model
    train (model)