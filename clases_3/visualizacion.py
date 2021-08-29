
import streamlit as st
import mysql.connector
import base64


# Source https://docs.streamlit.io/en/stable/tutorial/mysql.html#add-username-and-password-to-your-local-app-secrets


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


rows = run_query("SELECT * from results;")

# Print results.
for row in rows:
    # st.write({row[6][2:]})
    # """### Image from local file"""
    file_ = open("/mnt/c"+str(row[6][2:]), "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="waste" width=100 height=100>',
        unsafe_allow_html=True,
    )

    st.write(f"Prediction: {row[2]}")
    st.write(f"% Prediction: {row[3]}%")
    st.write(f"Recyle bag: {row[4]}")
    st.write(f"Real: {row[5]}")
    st.write(f" ")
