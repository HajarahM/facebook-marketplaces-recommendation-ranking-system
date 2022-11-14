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

if __name__ == '__main__':
    dataset = TextDataset()
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=1)

    print(dataset.num_classes)
    for batch, (data, labels) in enumerate(dataloader):
        print(data)
        print(labels)
        if batch == 0:
            break