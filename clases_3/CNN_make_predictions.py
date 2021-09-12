from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import mysql.connector
from mysql.connector import Error

from tkinter import *
from tkinter import filedialog

import time
import datetime

# These are the class labels from the training data
class_labels = [
    "Organic",
    "Recycle",
    "Trash"
]

# Load the json file that contains the model's structure
f = Path("//wsl$/Ubuntu-20.04/home/jduran/master-bigData/clasificadorImagenes/clases_3/model_structure_Entrenamiento_5_GCP_Septiembre_10_30k_3k_rescale.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights(
    "//wsl$/Ubuntu-20.04/home/jduran/master-bigData/clasificadorImagenes/clases_3/model_weights_C3_Entrenamiento_5_GCP_Septiembre_10_30k_3k_rescale.h5")


# load data "pruebas"
root = Tk()
root.filedir = filedialog.askdirectory(
    initialdir="C:/Users/jdura/Documents/pruebas",
    title="Select a directory image"
)

print(root.filedir)


datagen_test = ImageDataGenerator(rescale=1./255)
# root.mainloop()
data_pruebas = datagen_test.flow_from_directory(root.filedir,
                                                class_mode='categorical',
                                                target_size=(64, 64), batch_size=32,
                                                color_mode="rgb", shuffle=False)


files = data_pruebas.filenames
dataFiles = []
directory = root.filedir

for file in files:
    filePath = file.replace("\\", "/")
    dataFiles.append(directory+"/"+filePath)


x_p, y_p = data_pruebas.next()

print("Data generated")

results = model.predict(x_p)

print("Data predicted")

i = 0
fig = plt.figure(figsize=(20, 16))
columns = 6
rows = 6


try:
    connection = mysql.connector.connect(host='localhost',
                                         database='waste_classifier',
                                         user='rootx',
                                         password='Jdh910523',
                                         auth_plugin='mysql_native_password')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

        for result in results:
            most_likely_class_index = int(np.argmax(result))
            # Get the name of the most likely class
            class_label = class_labels[most_likely_class_index]
            class_likelihood = result[most_likely_class_index]*100

            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(
                ts).strftime('%Y-%m-%d %H:%M:%S')

            # Real classification
            realClassification = ''
            if(y_p[i][0] == 1.0):
                realClassification = "Organic"
            elif (y_p[i][1] == 1.0):
                realClassification = "Recycle"
            else:
                realClassification = "Trash"

            # Assigning recycling bag color
            recycleBagsColor = ''
            if(class_label == "Organic"):
                recycleBagsColor = "Green"
            elif (class_label == "Recycle"):
                recycleBagsColor = "White"
            else:
                recycleBagsColor = "Black"

            # Records to be insertnet into the row of MySQL
            records = [timestamp, class_label,
                       float(class_likelihood), recycleBagsColor, realClassification, dataFiles[i], "GCP5"]

            # Insert to DB
            cursor.execute(
                "INSERT INTO results (time, category_classified, percentage_prediction, recycle_bag_color, realClassification, path, model)  VALUES  (%s, %s, %s, %s, %s, %s, %s)",
                records)
            connection.commit()

            # Print the results
            print(
                "This is image is a {} - %: {:2f}".format(class_label, class_likelihood))
            print(cursor.rowcount, "Record inserted successfully into results table")

            # Figure for the classified elements
            fig.add_subplot(rows, columns, i+1)
            plt.gca().set_title("{} - %: {:2f}".format(class_label, class_likelihood), fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(x_p[i])

            # accumulator
            i += 1

        cursor.close()
        plt.show()


except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
