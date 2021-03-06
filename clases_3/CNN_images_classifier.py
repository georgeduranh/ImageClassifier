import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from pathlib import Path


RUN_NAME = "Entrenamiento_3_Junio_22"

# Load DATA
# create a data generator
datagen = ImageDataGenerator()

# load and iterate training dataset
data_train = datagen.flow_from_directory('/home/jduran/master-bigData/datos/datosProduccion3C/TRAIN/', class_mode='categorical',
                                         target_size=(64, 64), batch_size=64, seed=42, color_mode="rgb")
# load and iterate test dataset
data_test = datagen.flow_from_directory('/home/jduran/master-bigData/datos/datosProduccion3C/TEST/', class_mode='categorical',
                                        target_size=(64, 64), batch_size=64, seed=42, color_mode="rgb")

x_train, y_train = data_train.next()
x_test, y_test = data_test.next()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# # Normalize data set to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# # Convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, 3)
#y_test = keras.utils.to_categorical(y_test, 3 )
# print(y_test)

#
# # Create a model and add layers
model = Sequential()
#
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
model.add(Dense(3, activation="softmax"))
#
# Compile the model

opt = SGD(lr=0.1, momentum=0.9)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


logger = keras.callbacks.TensorBoard(
    log_dir='clases_3/logs/{}'.format(RUN_NAME),
    write_graph=True,
    histogram_freq=5
)


#model.fit_generator(x_train, steps_per_epoch=16, validation_data=x_test, validation_steps=8)

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=16,
    validation_data=(x_test, y_test),
    callbacks=[logger]
)

# Save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights_C3.h5")


model.summary()
