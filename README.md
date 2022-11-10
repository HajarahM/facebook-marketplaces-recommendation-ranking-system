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
