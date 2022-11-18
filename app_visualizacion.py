import streamlit as st
from src.maquetado import vistas


# change max message size which can be sent via websocket
# st.server.server_util.MESSAGE_SIZE_LIMIT = 300 * 1e6

st.set_page_config(page_title="Datastrofe")

st.set_option('deprecation.showPyplotGlobalUse', False)

# navigation links
link = st.sidebar.radio(label='Páginas', options=['Inicio', 'AED', 'Visualizacion', 'Informe', 'Machine Learning'])
vistas(link)

