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
        if not os.path.exists('./cleaned_images/'):
            os.makedirs('./cleaned_images/')
    except OSError:
        print ('Error: Creating directory. ' +  './cleaned_images/')

    #resize and save new image
    for n, item in enumerate(dirs[:12604], 1):
        im = Image.open(original_image_path + item)
        new_im = resize_image(final_size, im)
        old_file_path = os.path.splitext(f'./cleaned_images/{item}')[0]
        # new_im.save(f'./cleaned_images/{item}_resized.jpg')
        new_im.save(f'{old_file_path}_resized.jpg')

if __name__ == '__main__':    
    clean_image_data("./images/")
    