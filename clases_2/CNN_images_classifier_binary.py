import keras
from keras.backend import binary_crossentropy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from pathlib import Path


RUN_NAME = "Entrenamiento_9_imagenes_64x64_25epochs"

# Load DATA
# create a data generator
datagen = ImageDataGenerator()

# load and iterate training dataset
data_train = datagen.flow_from_directory('/home/jduran/master-bigData/datos/datosProduccion/TRAIN/', class_mode='categorical',
                                         target_size=(64, 64), batch_size=32)
# load and iterate test dataset
data_test = datagen.flow_from_directory('/home/jduran/master-bigData/datos/datosProduccion/TEST/', class_mode='categorical',
                                        target_size=(64, 64), batch_size=32)

x_train, y_train = data_train.next()
x_test, y_test = data_test.next()


# # Normalize data set to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#print(type(x_train))

#print(len(x_train))
#print(x_train)
#print(len(x_test))
#print(x_test)

### Resampling the data 
#x_total = np.append(x_train, x_test, axis=0)
#print(len(x_total))

# # Convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, 3)
#y_test = keras.utils.to_categorical(y_test, 3 )
# print(y_test)

#
# **********Create a model and add layers****************
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(
    64, 64, 3), activation="relu", name='Conv1'))
model.add(Conv2D(64, (3, 3), activation="relu", name='Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='Pooling1'))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', activation="relu", name='Conv3'))
model.add(Conv2D(128, (3, 3), activation="relu", name='Conv4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='Pooling2'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))


# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)


valData = 10

# Validation Data
#x_val = x_train[:valData]
#partial_x_train = x_train[valData:]

#y_val = y_train[:valData]
#partial_y_train = y_train[valData:]


# Saving the model logs
logger = keras.callbacks.TensorBoard(
    log_dir='/home/jduran/master-bigData/clasificadorImagenes/clases_2/logs/{}'.format(
        RUN_NAME),
    write_graph=True,
    histogram_freq=5
)


# # Train the model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=25,
    validation_data=(x_test, y_test),
    shuffle=True,
    callbacks=[logger]
)

# Save neural network structure
model_structure = model.to_json()
f = Path("/home/jduran/master-bigData/clasificadorImagenes/clases_2/model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights(
    "/home/jduran/master-bigData/clasificadorImagenes/clases_2/model_weights_C2.h5")


model.summary()



