from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
from torchvision.utils import save_image
from image_loader import create_folder
from torchvision.io import read_image
import torch
import torchvision
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
def repeat_channel(x):
            return x.repeat(3,1,1)

class ProcessImage:
    def __init__(self):
        
        #if not os.path.exists('test_image/image.jpg'):
        #    raise RuntimeError('Test image not found, upload image ...')
        #else:
        print('Loading image')

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

    def __call__(self, image):    
        #save image
        #create_folder('./test_image')
        #save_image(image, './test_image/image.jpg')  
        #display image
        #display_image = torchvision.io.read_image('image.jpg')
        #transforms.ToPILImage()(image)
        #process image
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)

        # Add a dimension to the image (from (batch_size, n_channels, height, width) to (n_channels, height, width).)
        image = image[None, :, :, :]
        print(type(image), image.shape)
        return image