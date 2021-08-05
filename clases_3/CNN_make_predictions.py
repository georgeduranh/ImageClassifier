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
    "Recycle",
    "Trash"
]

# Load the json file that contains the model's structure
f = Path("/home/jduran/master-bigData/clasificadorImagenes/clases_3/model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights(
    "/home/jduran/master-bigData/clasificadorImagenes/clases_3/model_weights_C3.h5")

datagen_test = ImageDataGenerator(rescale=1./255)
# load data "pruebas"
data_pruebas = datagen_test.flow_from_directory('/home/jduran/master-bigData/datos/pruebas/JD/',
                                                class_mode='categorical',
                                                target_size=(64, 64), batch_size=32,
                                                color_mode="rgb", shuffle=True)
x_p, y_p = data_pruebas.next()

print("Data genertion")

results = model.predict(x_p)
i = 0
fig = plt.figure(figsize=(20, 16))
columns = 6
rows = 6


try:
    connection = mysql.connector.connect(host='172.18.176.1',
                                         database='waste_classifier',
                                         user='root',
                                         password='Jdh910523')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

        # Pending to update the querie here *************************************************************************
        mySql_insert_query = """INSERT INTO results (idresults, time, category_classified, category_real, percentage_prediction) 
                           VALUES 
                           (2, '2021-07-26 15:00:00', 'organic', 'organic', '80') """

        cursor.execute(mySql_insert_query)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into results table")
        cursor.close()

        for result in results:
            most_likely_class_index = int(np.argmax(result))
            # Get the name of the most likely class
            class_label = class_labels[most_likely_class_index]
            class_likelihood = result[most_likely_class_index]*100

            # print(class_label)
            # print(class_likelihood)

            #fig.add_subplot(rows, columns, i+1)
            #plt.gca().set_title("{} - %: {:2f}".format(class_label, class_likelihood), fontsize=12)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.imshow(x_p[i])

            # Print the result
            #print("This is image is a {} - %: {:2f}".format(class_label, class_likelihood))
            i += 1

        plt.show()

except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
