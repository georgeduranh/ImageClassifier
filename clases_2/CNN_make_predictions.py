from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import mysql.connector
from mysql.connector import Error

# These are the class labels from the training data
class_labels = [
    "Organic",
    "Recycle"
]

# Load the json file that contains the model's structure
f = Path("/home/jduran/master-bigData/clasificadorImagenes/clases_2/model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights(
    "/home/jduran/master-bigData/clasificadorImagenes/clases_2/model_weights_C2.h5")

datagen_test = ImageDataGenerator(rescale=1./255)
# load data "pruebas"
data_pruebas = datagen_test.flow_from_directory('/home/jduran/master-bigData/datos/pruebas/JD/',
                                                class_mode='categorical',
                                                target_size=(64, 64), batch_size=32,  color_mode="rgb", shuffle=True)
x_p, y_p = data_pruebas.next()


results = model.predict(x_p)
i = 0
fig = plt.figure(figsize=(20, 16))
columns = 6
rows = 6

for result in results:
    most_likely_class_index = int(np.argmax(result))
    # Get the name of the most likely class
    class_label = class_labels[most_likely_class_index]
    class_likelihood = result[most_likely_class_index]*100

    fig.add_subplot(rows, columns, i+1)
    plt.gca().set_title("{} - %: {:2f}".format(class_label, class_likelihood), fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(x_p[i])
    # Print the result
    #print("This is image is a {} - %: {:2f}".format(class_label, class_likelihood))
    i += 1


plt.show()
