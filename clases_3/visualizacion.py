
import pandas as pd
from streamlit import caching
from matplotlib.widgets import Slider, CheckButtons
import streamlit as st
import mysql.connector
import base64
import streamlit.components.v1 as components
import os
import time
import datetime
import seaborn as sns

# Predictions
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# Data base
import mysql.connector
from mysql.connector import Error


# Source https://docs.streamlit.io/en/stable/tutorial/mysql.html#add-username-and-password-to-your-local-app-secrets
# st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return mysql.connector.connect(**st.secrets["mysql"])


conn = init_connection()

# Perform query.
# Uses st.cache to only rerun when the query changes or after 10 min.


@st.cache(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def runQuery():
    caching.clear_cache()
    rowsx = run_query(
        "SELECT category_classified, COUNT(category_classified) from results group by category_classified Order by COUNT(category_classified) DESC;")
    return rowsx


# Montlhy plot
# if st.sidebar.checkbox("Análisis por año y mes", True, key=1):

month = run_query(
    "SELECT YEAR(time), MONTH(time), category_classified, COUNT(category_classified)  from results group by  YEAR(time), MONTH(time), category_classified  order BY category_classified DESC;")
df = pd.DataFrame(month, columns=['Año', 'Mes', 'Categoria', 'Cantidad'])
st.sidebar.checkbox("Análisis por año y mes", True, key=1)
selectYear = st.sidebar.selectbox(
    'Selecciona el año:', df['Año'].drop_duplicates())
selectMonth = st.sidebar.selectbox(
    'Selecciona el mes:', df['Mes'].drop_duplicates())

yearData = df[df['Año'] == selectYear]
monthData = df[yearData['Mes'] == selectMonth]

print("DF month")
print(monthData)


# Bar plot
fig2, ax = plt.subplots()
sns.barplot(data=monthData, y='Cantidad', x='Mes', hue='Categoria',
            palette=dict(Organico="Green", Reciclable="Gray", NoAprovechable="Black"))
st.pyplot(fig2)


# Image loading
image_file = st.file_uploader("Selecciona las imagenes de residuos que deseas clasificar:",
                              accept_multiple_files=True, help="Máximo tamaño 20Mb", key="25")

col1, col2, = st.columns(2)

if st.sidebar.button('Recargar datos'):
    runQuery()

# Button for saving images
if col1.button('1. Guardar imagenes'):
    ts = time.time()
    dir = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    print(dir)
    if not os.path.exists("clases_3/images_loaded/"+dir):
        os.mkdir("clases_3/images_loaded/"+dir)

    for picture in image_file:
        if picture is not None:
            file_details = {"FileName": picture.name,
                            "FileType": picture.type}
        # st.write(file_details)
        with open(os.path.join("clases_3/images_loaded/"+dir, picture.name), "wb") as f:
            f.write(picture.getbuffer())

    st.success("Archivos guardados")

# Button for classifing images
if col2.button('2. Clasificar imágenes'):
    # These are the class labels from the training data
    class_labels = [
        "Organico",
        "Reciclable",
        "NoAprovechable"
    ]

    # Load the json file that contains the model's structure
    f = Path("/home/jduran/master-bigData/clasificadorImagenes/clases_3/model_structure_Entrenamiento_5_GCP_Septiembre_10_30k_3k_rescale.json")
    model_structure = f.read_text()

    # Recreate the Keras model object from the json data
    model = model_from_json(model_structure)

    # Re-load the model's trained weights
    model.load_weights(
        "/home/jduran/master-bigData/clasificadorImagenes/clases_3/model_weights_C3_Entrenamiento_5_GCP_Septiembre_10_30k_3k_rescale.h5")

    print(image_file)

    ts = time.time()
    dirHoy = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    for picture in image_file:
        if picture is not None:
            directorio = "/home/jduran/master-bigData/clasificadorImagenes/clases_3/images_loaded/" + \
                str(dirHoy)+"/"
            img = image.load_img(directorio+picture.name, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            results = model.predict(x)

            print("Data predicted")

            try:
                connection = mysql.connector.connect(host='172.18.240.1',
                                                     database='waste_classifier',
                                                     user='rootall01',
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
                        dir = datetime.datetime.fromtimestamp(
                            ts).strftime('%Y-%m-%d')

                        # Assigning recycling bag color
                        recycleBagsColor = ''
                        if(class_label == "Organico"):
                            recycleBagsColor = "Verde"
                        elif (class_label == "Reciclable"):
                            recycleBagsColor = "Blanco"
                        else:
                            recycleBagsColor = "Negro"

                        # Records to be insertnet into the row of MySQL
                        records = [timestamp, class_label,
                                   float(class_likelihood), recycleBagsColor, "/home/jduran/master-bigData/clasificadorImagenes/clases_3/images_loaded/"+dir+"/"+picture.name, "GCP5"]

                        # Insert to DB
                        cursor.execute(
                            "INSERT INTO results (time, category_classified, percentage_prediction, recycle_bag_color, path, model)  VALUES  (%s, %s, %s, %s, %s, %s)",
                            records)
                        connection.commit()

                        # Print the results
                        print(
                            "This is image is a {} - %: {:2f}".format(class_label, class_likelihood))
                        print(cursor.rowcount,
                              "Record inserted successfully into results table")

                    cursor.close()

            except Error as e:
                print("Error while connecting to MySQL", e)
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()
                    print("MySQL connection is closed")

rows = run_query("SELECT * from results order by idresults DESC;")

# Print results in streamlit.
for row in rows:
    file_ = open(str(row[5][0:]), "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    col1, col2 = st.columns(2)

    col1.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="waste" width=120 height=100>',
        unsafe_allow_html=True,
    )

    col2.write(
        f"Fecha y hora: {row[1]} Clasificación: {row[2]}, Porcentaje: {row[3]}%, Bolsa: {row[4]} ")
    st.write(f" ")
