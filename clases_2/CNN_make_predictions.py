from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

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

# Load an image file to test, resizing it to 64x64 pixels 
img = image.load_img(
    "/home/jduran/master-bigData/datos/pruebas/JD/R/R (1).jpeg", target_size=(64, 64))


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
