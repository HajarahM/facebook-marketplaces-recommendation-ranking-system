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
```
## Exploring the Dataset
Following addition to the 'Images.csv' and 'Products.csv' files to the repository and importing of the images folder, I then cleaned the dataset. 

### Cleaning the Tabular Dataset
The file 'clean_tabular_data.py' was created containing code to clean the data, specifically converting the text data to numerical data where needed and to delete rows with missing information.
Sample code below;
```python
    #convert colomn datatypes
    df_products = pd.read_csv("Products.csv", index_col=0, lineterminator='\n')
    #price
    df_products['price'] = df_products['price'].replace('[\£,]', '', regex=True).astype(float)
    df_products['price'] = pd.to_numeric(df_products['price'])
    #location
    df_products['location'] = df_products['location'].astype('category')

    print(df_products)
    df_products['location'].describe()

    #drop Rows with missing value / NaN in any column
    clean_df_products = df_products.dropna()
    print("Modified Dataframe : ")
    print(clean_df_products)
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
        im = im.resize(new_image_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return new_im

    def clean_image_data(original_image_path):
        dirs = os.listdir(original_image_path)
        final_size = 256

        #create 'cleaned_images' folder
        try:
            if not os.path.exists('./cleaned_images/'):
                os.makedirs('./cleaned_images/')
        except OSError:
            print ('Error: Creating directory. ' +  './cleaned_images/')

        #resize and save new image
        for n, item in enumerate(dirs[:5], 1):
            im = Image.open(original_image_path + item)
            new_im = resize_image(final_size, im)
            new_im.save(f'./cleaned_images/{n}_resized.jpg')

    if __name__ == '__main__':    
        clean_image_data("./images/")
```      