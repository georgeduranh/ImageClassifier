
import streamlit as st
import mysql.connector
import base64
import streamlit.components.v1 as components


# Source https://docs.streamlit.io/en/stable/tutorial/mysql.html#add-username-and-password-to-your-local-app-secrets

st.set_page_config(layout="wide")


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
    height=850,
)


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

    st.write(
        f"Prediction: {row[2]} --- % Prediction: {row[3]}% --- Bag color: {row[4]} --- Real: {row[5]}")
    #st.write(f"% Prediction: {row[3]}%")
    #st.write(f"Recyle bag: {row[4]}")
    #st.write(f"Real: {row[5]}")
    st.write(f" ")
