
from streamlit import caching
from matplotlib.widgets import Slider, CheckButtons
import streamlit as st
import mysql.connector
import base64
import streamlit.components.v1 as components
import os
import time
import datetime

# Predictions
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


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


@st.cache(ttl=6)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


# bootstrap 4 collapse example
components.html(
    """
    <div class='tableauPlaceholder' id='viz1630280517846' style='position: relative'><noscript><a href='#'>
    <img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;re&#47;reporte_clasificacion&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a>
    </noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> 
    <param name='site_root' value='' /><param name='name' value='reporte_clasificacion&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' />
    <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;re&#47;reporte_clasificacion&#47;Dashboard1&#47;1.png' /> 
    <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' />
    <param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' />
    </object></div>                <script type='text/javascript'>                  
      var divElement = document.getElementById('viz1630280517846');                
          var vizElement = divElement.getElementsByTagName('object')[0];                  
            if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1027px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """,
    height=600,
)


def runQuery():
    caching.clear_cache()
    rowsx = run_query(
        "SELECT category_classified, COUNT(category_classified) from results group by category_classified Order by COUNT(category_classified) DESC;")
    return rowsx

image_file = st.file_uploader("Selecciona las imagenes de residuos que deseas clasificar:",
                              accept_multiple_files=True, help="Máximo tamaño 20Mb", key="25")

col1, col2, col3 = st.columns(3)
if col1.button('Recargar'):
    runQuery()

rowsx = dict(runQuery())
print((rowsx))

fig = plt.figure()
plt.bar(list(rowsx.keys()), list(rowsx.values()), align='center')
st.pyplot(fig)


if col2.button('Guardar'):
    ts = time.time()
    dir = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    if not os.path.exists("images_loaded/"+dir):
        os.mkdir("images_loaded/"+dir)

    for picture in image_file:
        if picture is not None:
            file_details = {"FileName": picture.name,
                            "FileType": picture.type}
        # st.write(file_details)
        with open(os.path.join("images_loaded/"+dir, picture.name), "wb") as f:
            f.write(picture.getbuffer())

    st.success("Archivos guardados")

if col3.button('Clasificar'):
    # These are the class labels from the training data
    class_labels = [
        "Organic",
        "Recycle",
        "Trash"
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
    for picture in image_file:
        if picture is not None:
            img = image.load_img(
                "/home/jduran/master-bigData/clasificadorImagenes/clases_3/images_loaded/2021-09-11/"+picture.name, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            results = model.predict(x)

            print("Data predicted")

            try:
                connection = mysql.connector.connect(host='172.27.224.1',
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
                                   float(class_likelihood), recycleBagsColor, "NA", "/home/jduran/master-bigData/clasificadorImagenes/clases_3/images_loaded/2021-09-11/"+picture.name, "GCP5"]

                        # Insert to DB
                        cursor.execute(
                            "INSERT INTO results (time, category_classified, percentage_prediction, recycle_bag_color, realClassification, path, model)  VALUES  (%s, %s, %s, %s, %s, %s, %s)",
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
        file_ = open(str(row[6][0:]), "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="waste" width=120 height=100>',
            unsafe_allow_html=True,
        )

        st.write(
            f"Time: {row[1]} --- Prediction: {row[2]} --- % Prediction: {row[3]}% --- Bag color: {row[4]} ")
        st.write(f" ")


