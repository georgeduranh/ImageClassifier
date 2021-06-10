from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "Organic",
    "Recycle",
    "Trash"
]

# Load the json file that contains the model's structure
f = Path("model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_weights_C3.h5")

# Load an image file to test, resizing it to 32x32 pixels (as required by this model)
img = image.load_img(
    "/home/jduran/master-bigData/datos/pruebas/trash46.jpg", target_size=(32, 32))


# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
list_of_images = np.expand_dims(img, axis=0)

# Make a prediction using the model
results = model.predict(list_of_images)

# Since we are only testing one image, we only need to check the first result
single_result = results[0]

# We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

# Get the name of the most likely class
class_label = class_labels[most_likely_class_index]

# Print the result
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))
