import streamlit as st
from src.control_graficos import control_graficos
from src.carga_datos import load_dataframe
from streamlit.components.v1 import iframe

def vistas(link):
    """
    Función de redirección entre páginas
    :param link: str, opción seleccionada en el radio button
    :return:
    """
    if link == 'Visualizacion':
        st.header("Bienvenido al entorno de visualización")
        st.subheader("Inicio rápido")
        st.markdown("Para visualizar los datos, primero suba el archivo con los mismos, "
                    "seleccione un tema, luego el tipo de gráfico "
                    "y los parámetros de éste. Por último, puede descargar el gráfico mostrado.")

        st.sidebar.subheader('Opciones')

        st.sidebar.subheader("Importe los datos")
        uploaded_file = st.sidebar.file_uploader(label="Importe aquí el archivo csv o excel.",
                                                 accept_multiple_files=False,
                                                 type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df, columns = load_dataframe(uploaded_file=uploaded_file)

            st.sidebar.subheader("Visualización de los datos")

            show_data = st.sidebar.checkbox(label='Mostrar datos')

            if show_data:
                try:
                    st.subheader("Visualización")
                    number_of_rows = st.sidebar.number_input(label='Seleccione el número de filas', min_value=2)

                    st.dataframe(df.head(number_of_rows))
                except Exception as e:
                    print(e)

            st.sidebar.subheader("Selección de tema")

            theme_selection = st.sidebar.selectbox(label="Seleccione el tema",
                                                   options=['plotly', 'plotly_white',
                                                            'ggplot2',
                                                            'seaborn', 'simple_white'])
            st.sidebar.subheader("Selección de gráfico")
            chart_type = st.sidebar.selectbox(label="Seleccione el tipo de gráfico.",
                                              options=['Grafico de dispersion', 'Grafico por sectores',
                                                        'Histograma', 'Grafico de lineas', 'Grafico de barras'])
                                                       #'Density contour',
                                                       #'Sunburst','Pie Charts','Density heatmaps',
                                                       #'Histogram', 'Box plots','Tree maps',
                                                       #'Violin plots', ])

            control_graficos(chart_type=chart_type, df=df, dropdown_options=columns, template=theme_selection)

    if link == 'Referencias':
        st.header('Referencias')
        st.write("Este entorno está basado en el proyecto Open Source OpenCharts.")
        st.subheader('Repositorio de la aplicación')
        st.write("El código de esta aplicación puede consultarse en el repositorio: https://github.com/alu0100898293/ADM-Visualizacion")
        st.subheader('Tutorial de desarrollo')
        st.write("Para el desarrollo de este entorno se han llevado a cabo las indicaciones marcadas por los vídeos del usuario de Youtube" 
            "The Fullstack Ninja en la colección https://www.youtube.com/playlist?list=PLgf5tk2HvlhONM16aLWjhdJPxRptglWdW")
        st.subheader('Repositorio de OpenCharts')
        st.write("Puede consultar el repositorio del proyecto OpenCharts en el repositorio https://github.com/bodealamu/opencharts")