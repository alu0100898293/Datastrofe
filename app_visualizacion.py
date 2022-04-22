import streamlit as st
from src.maquetado import vistas


# change max message size which can be sent via websocket
st.server.server_util.MESSAGE_SIZE_LIMIT = 300 * 1e6

# navigation links
link = st.sidebar.radio(label='Páginas', options=['Visualizacion', 'Referencias'])
vistas(link)

