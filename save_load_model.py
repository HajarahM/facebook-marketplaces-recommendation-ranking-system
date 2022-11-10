import os
import torch
from pathlib import Path, PosixPath
from datetime import datetime
from image_loader import model
from image_loader import train
from image_loader import CNN


def create_date_directory(path : str) -> PosixPath:

    now = datetime.today()
    nTime = now.strftime("%Y-%m-%d_%H-%M-%S")

    if not Path(path).joinpath(nTime).exists():
      Path(path).joinpath(nTime).mkdir(parents=True)
    
    return Path(path).joinpath(nTime)

def create_folder(directory):
        """ 
        Description
        -----------
        Creates a new folder in the specified directory if it doesn't already exist. Incase of an OS-Error, an error message is printed out.
        
        Parameters
        ----------
        directory: str, the path to the directory where the new file is to be saved. "./" being the current folder of the python file.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

#create dated directory and weights folder
date_directory = create_date_directory('./model_evaluation')
create_folder(f'./model_evaluation/{date_directory}/weights')

def save_model():
    torch.save(
        {'epoch': train.epoch,
        'model_state_dict': model.state_dict(),
        'loss': train.loss},
        f'./model_evaluation/{date_directory}/weights/model.pt')

def load_model():
    model = CNN()
    train(model)
    checkpoint = torch.load(f'./model_evaluation/{date_directory}/weights/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    train.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train.epoch = checkpoint['epoch']
    train.loss = checkpoint['loss']