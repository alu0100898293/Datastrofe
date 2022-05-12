import streamlit as st
from src.control_graficos import control_graficos
from src.carga_datos import load_dataframe
from src.informe import mostrar_informe
from streamlit.components.v1 import iframe

def vistas(link):
    """
    Función de redirección entre páginas
    :param link: str, opción seleccionada en el radio button
    :return:
    """
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
    else:    
        st.header("Bienvenido al maravilloso mundo del Análisis de Datos")
        st.subheader("Inicio rápido")
        st.markdown("Para comenzar con los análisis de datos, el primer paso es "
                    "cargar el conjunto de datos que vamos a estudiar. "
                    "Así que importa los datos y elige la función que quieras emplear. "
                    "Si marca la casilla de limpiar datos, se borrarán las filas duplciadas y "
                    "aquellas en las que falten valores.")

        st.sidebar.subheader('Opciones')

        clean_data = st.sidebar.checkbox(label='Limpiar datos')

        st.sidebar.subheader("Importe los datos")
        uploaded_file = st.sidebar.file_uploader(label="Importe aquí el archivo csv o excel.",
                                                accept_multiple_files=False,
                                                type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df, columns = load_dataframe(uploaded_file=uploaded_file, clean_data=clean_data)

            st.sidebar.subheader("Mostrar el dataset")

            show_data = st.sidebar.checkbox(label='Mostrar datos')

            if show_data:
                try:
                    st.subheader("Dataset importado")
                    number_of_rows = st.sidebar.number_input(label='Seleccione el número de filas', min_value=2)

                    st.dataframe(df.head(number_of_rows))
                except Exception as e:
                    print(e)

            if link == 'Visualizacion':
                st.subheader("Bienvenido al entorno de visualización")
                st.markdown("La visualización de datos dispone de varios temas a elegir, "
                            "así como múltiples tipos de gráficos que puede adaptar según "
                            "guste mediante parámetros diversos. De forma adicional, puede descargar el gráfico mostrado.")

                st.sidebar.subheader("Selección de tema")

                theme_selection = st.sidebar.selectbox(label="Seleccione el tema",
                                                    options=['plotly', 'plotly_white',
                                                                'ggplot2',
                                                                'seaborn', 'simple_white'])
                st.sidebar.subheader("Selección de gráfico")
                chart_type = st.sidebar.selectbox(label="Seleccione el tipo de gráfico.",
                                                options=['Grafico de dispersion', 'Grafico por sectores',
                                                            'Histograma', 'Grafico de lineas', 'Grafico de barras',
                                                            'Grafico de violin', 'Grafico de cajas'])
                                                        #'Density contour',
                                                        #'Sunburst','Pie Charts','Density heatmaps',
                                                        #'Tree maps',])

                control_graficos(chart_type=chart_type, df=df, dropdown_options=columns, template=theme_selection)
            
            if link == 'Informe':
                st.subheader("Información del dataset")
                st.markdown("A continuación se muestra un informe general sobre los datos importados, "
                            "tenga en cuenta que la creación de este informe no está recomendado para "
                            "Datasets de gran tamaño debido a la excesiva duración del proceso.")

                mostrar_informe(df=df) 
    