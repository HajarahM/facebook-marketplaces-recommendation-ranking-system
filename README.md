# Facebook Marketplace Recommendation Ranking System

## Setting up the Environment
After setting up the Github repo 'Facebook Marketplace Recommendation Ranking System', I created the virtual environment for this project and saved it as a directory 'env' in this repository, then added it to the .gitignore to reduce on the size of the repo due to max-size limitations.
I then created a requirements.txt file to list all the required installations
```python
    pandas
    numpy
    Pillow
    matplotlib
    imagesize
    sklearn
    torch
    torchvision
    transformers
    scipy
    functions
    scikit-learn
```
## Exploring the Dataset
Following addition to the 'Images.csv' and 'Products.csv' files to the repository and importing of the images folder, I then cleaned the dataset. 

### Cleaning the Tabular Dataset
The file 'clean_tabular_data.py' was created containing code to clean the data, specifically converting the text data to numerical data where needed and to delete rows with missing information.
Sample code below;
```python
    import pandas as pd

    #import csv files as dataframes
    products_df = pd.read_csv('Products.csv', lineterminator='\n')
    images_df = pd.read_csv('Images.csv', lineterminator='\n')

    #rename 'id' columns
    df = products_df.rename(columns={'id':'product_id'})
    idf = images_df.rename(columns={'id':'image_id'})

    #merge the 2 tables
    combined_df = pd.merge(df, idf, how="inner", on=["product_id"])
    #delete unnamed columns (index columns from initial tables)
    combined_df.drop(combined_df.columns[combined_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

    #change datatype of price column to float, remove currency sign, remove all emojis and make numeric
    combined_df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    combined_df['price'] = combined_df['price'].replace('[\£,]', '', regex=True).astype(float)
    combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')

    #clean product names - strip all text after first "|"
    combined_df['product_name'] = combined_df['product_name'].str.split('|').str[0]

    #split category column into main and sub categories
    combined_df['main_category'], combined_df['sub_category'] = combined_df['category'].str.split('/',1).str

    #Specify Category columns
    combined_df['category'] = combined_df['category'].astype('category')
    combined_df['location'] = combined_df['location'].astype('category')

    #delete rows with empty data, missing values
    cleaned_df = combined_df.dropna()

    #print statements
    print(f'Products dataset: {len(df)}')
    print(f'Image dataset {len(idf)}')
    print(f'Combined products dataframe: {len(combined_df)}')
    print(f'Cleaned products dataframe: {len(cleaned_df)} some products have mulitple images')
    print(cleaned_df.head())

    # save cleaned_products dataframe to csv file
    cleaned_df.to_csv('cleaned_products.csv')
```
### Cleaning the Image Dataset
#### Analyzing the Images
The file 'analyze_images.py' was created containing code to analyze the multiple dimensions/size of the images by plotting them for visual presentation
Sample code below;
```python
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path
    import imagesize
    import numpy as np
    # Get the Image Resolutions
    imgs = [img.name for img in Path('./images/').iterdir() if img.suffix == ".jpg"]
    img_meta = {}
    for f in imgs: img_meta[str(f)] = imagesize.get('./images/'+f)
    # Convert it to Dataframe and compute aspect ratio
    img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', copy=False)
    img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
    img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)
    print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
    img_meta_df.head()
    # Visualize Image Resolutions
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    points = ax.scatter(img_meta_df.Width, img_meta_df.Height, color='blue', alpha=0.5, s=img_meta_df["Aspect Ratio"]*100, picker=True)
    ax.set_title("Image Resolution")
    ax.set_xlabel("Width", size=14)
    ax.set_ylabel("Height", size=14)
    fig = plt.show()
```
#### Cleaning the images
The file 'clean_images.py' was created containing code to clean the images. A pipeline was created to take in a filepath of the folder which contains the images, then clean them (change all images to the same image size of 256) and save them into a new folder called "cleaned_images".
Sample code below;
```python
from PIL import Image
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def clean_image_data(original_image_path):
    dirs = os.listdir(original_image_path)
    final_size = 256

    #create 'cleaned_images' folder
    try:
        if not os.path.exists('./clean_images/'):
            os.makedirs('./clean_images/')
    except OSError:
        print ('Error: Creating directory. ' +  './clean_images/')

    #resize and save new image
    for n, item in enumerate(dirs[:126040], 1):
        im = Image.open(original_image_path + item)
        new_im = resize_image(final_size, im)
        # old_file_path = os.path.splitext(f'./cleaned_images/{item}')[0]
        new_im.save(f'./clean_images/{item}')
        # new_im.save(f'{old_file_path}.jpg')

if __name__ == '__main__':
    clean_image_data("./images/")
```      
## Create Vision Model
I followed the following steps to create a Vision model - training it using Facebook Marketplace pictures to predict the image categories.
    1. Created a Pytorch dataset and built a CNN (Convolutional Neural Network) model with a training loop 
    2. Fine-tuned using a pre-trained model
    3. Saved weights of each epoch by date and final model
    4. Created an image processor script to get user input of image which will be used by trained model to predict it's category
The breakdown of the detail steps and respective code follow below

### Task 1 - Pytorch Dataset
To start with, I created a Pytorch dataset from the Facebook Marketplace products csv cleaned file that would then feed entries into the classification model.
During the cleaning of the two csv files (products csv and images csv), I had merged two to link the image ID to product details, especially to know what image belongs to what category.
The class for this torch.utils.data.dataset dataset is named 'ProductsDataset' and follows the following steps
    1. Verify that the 'clean_images' folder exists, if not it throws an error
    2. Otherwise, if the 'clean_images' folder exists, then program continues to load the 'cleaned_images' csv as a pandas dataframe ('self.products)
    3.From the products dataframe, the colomns for 'main_category' and 'descriptions' are defined as 'self.labels' and self.descriptions' respectively. And number of classes is obtained by getting the length of 'self.labels'.
    4. The source to be used for naming the files is also defined from 'image_id' colomn as self.files.
    5. To assign a label to each category, I used an encoder as shown in the code below and also defined the respective decoder that will be used to display the description of the predicted category.
    6. I then used image transformations to resize, center, crop, turn to Tensor and normalize with specified mean and standard deviation for both RGB and grey images.
sample code below;
``` python
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
    def __len__(self):
        return len(self.files)
```  
### Task 2 - Building a CNN - Convolutional Neural Network
Next, I split the datasets to have a training dataset and a validation dataset using the code below
```python
def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = random_split(dataset, [train_split, len(dataset) - train_split])
    return train_dataset, validation_dataset
``` 
I then created a CNN class, first from scratch defining each step with the following layers ...
```python
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
``` 
This resulted in an accuracy of average 0.2

### Task 3 - Buidling the training loop
I then defined a 'train' function which takes in the model as its 1st positional argument as well as number of epochs it will train for.
In this function, there is a loop interating through the number of batches(to be specified) of the dataset while updating the model's paramenters. It loops through the entire dataset as many times as the number of epochs.
The loss is printed after every prediction and graphically displayed on a TensorBoard.
``` python
def train(model, epochs=20):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001) #lr is learning rate
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
``` 

### Task 4 - Fine-tuning a pre-trained model
To obtain an even better accuracy and loss, I leveraged on work already done by others and used a pre-trained model off the shelf through 'transfer-learning'.
The pre-trained model is 'ResNet-50' which clarifies the images passed using the previously created dataset. To dop this, I replaced the final linear layer of the model with another linear layer whose outpt size is the same as the number of categories.
Sample code below;
``` python
def train(model, epochs=5):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
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
            Accuracy = round(torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels), 2)
            writer.add_scalar('Accuracy', Accuracy, batch_idx)
            batch_idx += 1            
            Losses = round(loss.item(), 2)
            p_bar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {Accuracy}, Losses = {Losses}")
```
### Task 5 - Saving the weights whilst training
Within the epoch training loop, I created a folder named 'model_evaluation' and within this programmatically created a time-stamed folder (date+hour) for each model by epoch number within a sub-folder called 'wesights'. Due to the potential massive size of parameters, this folder was added to 'gitignore file.
```python
#create dated directory and weights folder
        date_directory_path = create_date_directory('./model_evaluation')
        create_folder(f'{date_directory_path}/weights/{epoch+1}')
        # training loop to save the weights of the model at the end of every epoch.
        torch.save(
            {'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'accuracy': Accuracy,
            'loss': loss},
            f'{date_directory_path}/weights/epoch_{epoch+1}/model.pt')
```
### Task 6 - Saving the final model
After establishing the significant accuracy of over 80% (results below) I then saved the model into a 'final_models' folder saved as image_model.pt
```python
    create_folder('./final_models')
    torch.save(model.state_dict(),'final_models/image_model.pt')
    #save pickle file of decoder dictionary
    pickle.dump(dataset.decoder, open('image_decoder.pkl', 'wb'))
```
Model Results (5 epochs with 16 batches each)
____________________________________________________________________________________
Epoch = 1/5. Acc = 0.44, Losses = 1.57: 100%| 787/787
Epoch = 2/5. Acc = 0.5, Losses = 1.81: 100%| 787/787
Epoch = 3/5. Acc = 0.69, Losses = 0.7: 100%| 787/787
Epoch = 4/5. Acc = 0.88, Losses = 0.68: 100%|787/787 
Epoch = 5/5. Acc = 0.88, Losses = 0.58: 100%|787/787
_________________________________________________________________________________________
![alt text](https://github.com/HajarahM/facebook-marketplaces-recommendation-ranking-system/blob/main/README%20images/image_model_results.png?raw=true)

### Task 7 - Creating an image processor script
Finally I created an image processor script (image_processor.py) that would take in an image and apply the transformations needed (in Task 1) to be fed to the model. 
I added the dimension to the beginning of the image to make it a batch-size of 1, added code to save the image obtained from user to a 'test file' from where to would be otained for processing.
Sample code below;
```python
class ProcessImage:
    def __init__(self):        
        if not os.path.exists('test_image/image.jpg'):
            raise RuntimeError('Test image not found, upload image ...')
        else:
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
    def __call__(self):    
        #save image
        create_folder('./test_image')
        save_image(image, './test_image/image.jpg')  
        #display image
        display_image = torchvision.io.read_image('image.jpg')
        transforms.ToPILImage()(display_image)
        #process image
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)
        # Add a dimension to the image (from (batch_size, n_channels, height, width) to (n_channels, height, width).)
        image = image[1, :, :, :]
        print(type(image), image.shape)
        return image 
```