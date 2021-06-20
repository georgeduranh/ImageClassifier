
# source https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


import os
import os.path


# path joining version for other paths
# Change TRAIN by TEST to perform data augmentation over TEST dataset
DIR = '/home/jduran/master-bigData/datos/datosProduccion3C/TRAIN/T/'
archivos = (len([name for name in os.listdir(DIR)
            if os.path.isfile(os.path.join(DIR, name))]))


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
    file = "/home/jduran/master-bigData/datos/datosProduccion3C/TRAIN/T/TO_ ("+str(
        i)+").jpg"
    # this is a PIL image
    img = load_img(file)
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    # this is a Numpy array with shape (1, 3, 150, 150)
    x = x.reshape((1,) + x.shape)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    j = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='/home/jduran/master-bigData/datos/datosProduccion3C/TRAIN/T', save_prefix='Tx', save_format='jpg'):
        j += 1
        if j > 9:
            break  # otherwise the generator would loop indefinitely
