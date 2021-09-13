
# Referencias
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


import os
import os.path


# Source of image files
# Change TRAIN by TEST to perform data augmentation over TEST dataset
DIR = '/home/jduran/master-bigData/datos/Modelo 2/TEST/T/'
archivos = (len([name for name in os.listdir(DIR)
            if os.path.isfile(os.path.join(DIR, name))]))

#Generation of the images with different angles 
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Data augmentation over each file in the path

for i in range(1, archivos, 1):
    file = "/home/jduran/master-bigData/datos/Modelo 2/TEST/T/T_ ("+str(
        i)+").jpg"
    # this is a PIL image
    img = load_img(file)
    x = img_to_array(img)  # this is a Numpy array
    x = x.reshape((1,) + x.shape)

    #Saving the results
    j = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='/home/jduran/master-bigData/datos/Modelo 2/TEST/T/', save_prefix='Tx', save_format='jpg'):
        j += 1
        if j > 7:  ##number of data augmentation
            break  # otherwise the generator would loop indefinitely
