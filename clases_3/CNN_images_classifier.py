import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from pathlib import Path


# Name of logs
RUN_NAME = "Entrenamiento_2_Local_Agosto_22_15k_data"

# Load DATA
# https://stackoverflow.com/questions/42868982/how-do-i-check-the-order-in-which-keras-flow-from-directory-method-processes-fo

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/home/jduran/master-bigData/datos/Modelo 2/Sampling/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '/home/jduran/master-bigData/datos/Modelo 2/TEST/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')


""" # create a data generator
datagen = ImageDataGenerator()

# load and iterate training dataset
data_train = datagen.flow_from_directory('/home/jduran/master-bigData/datos/Modelo 2/Sampling/', class_mode='categorical',
                                         target_size=(64, 64), batch_size=15710, seed=42, color_mode="rgb")
# load and iterate test dataset
data_test = datagen.flow_from_directory('/home/jduran/master-bigData/datos/Modelo 2/TEST/', class_mode='categorical',
                                        target_size=(64, 64), batch_size=3572, seed=42, color_mode="rgb")

x_train, y_train = data_train.next()
x_test, y_test = data_test.next()



print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# # Normalize data set to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

#Validation data 
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train =  y_train[10000:]

print(y_val.shape)
print(partial_y_train.shape)
print(x_val.shape)
print(partial_x_train.shape) """


#
# # Create a model and add layers
# # Create a model and add layers
model = Sequential()
#
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=(64, 64, 3), activation="relu", name='Conv1'))
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


# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

logger = keras.callbacks.TensorBoard(
    log_dir='/home/jduran/master-bigData/clasificadorImagenes/clases_3/logs/{}'.format(
        RUN_NAME),
    write_graph=True,
    histogram_freq=5
)

#model.fit_generator(x_train, steps_per_epoch=16, validation_data=x_test, validation_steps=8)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=300,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50)

""" # Train the model
history= model.fit(
          partial_x_train,
          partial_y_train,
          batch_size=8,
          epochs=80,
          validation_data=(x_val, y_val),
          shuffle=False,
          callbacks=[logger]
      ) """


# plot the model accuracy and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'validation accuracy'])


# plot the model accuracy and validation accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'validation loss'])


# Save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights_C3.h5")


model.summary()


# model.evaluate(x_test,y_test)
