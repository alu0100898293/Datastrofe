import streamlit as st
from src.AED import analisis_exploratorio
from src.control_graficos import control_graficos
from src.carga_datos import load_dataframe
from src.informe import mostrar_informe
from src.machine_learning import ml_selector
from PIL import Image

def vistas(link):
    """
    Función de redirección entre páginas
    :param link: str, opción seleccionada en el radio button
    :return:
    """
    image = Image.open('imgs/logo.png')
    st.image(image)
    if link == 'Inicio':
        st.header('Bienvenidos al maravilloso mundo del anáilisis de datos')
        st.markdown("Adéntrate en el maravilloso mundo del anáilisis de datos con "
                    "nuestro entorno gracias a sus mecánicas de visualización y aprendizaje "
                    "automático. Emplea las vistas del menú lateral para empezar. ")

        st.markdown("A modo de recomendación, puedes descargar el Dataset de _Star Classification_ en el "
                    "siguiente [enlace](https://drive.google.com/file/d/1jQHUQAZc002zDfbQXtD-X9VHdJ5cNbpo/view?usp=sharing)")

        st.header('Referencias')
        st.write("Este entorno está basado en el proyecto Open Source OpenCharts.")
        st.markdown('#### Repositorio de la aplicación')
        st.markdown("El código de esta aplicación puede consultarse en el repositorio: https://github.com/alu0100898293/Datastrofe")
        st.markdown('#### Tutorial de desarrollo')
        st.markdown("Para el desarrollo de este entorno se han llevado a cabo las indicaciones marcadas por los vídeos del usuario de Youtube" 
            "The Fullstack Ninja en la colección: https://www.youtube.com/playlist?list=PLgf5tk2HvlhONM16aLWjhdJPxRptglWdW")
        st.markdown('#### Repositorio de OpenCharts')
        st.markdown("Puede consultar el repositorio del proyecto OpenCharts en el repositorio: https://github.com/bodealamu/opencharts")
    else:    
        st.subheader("Inicio rápido")
        st.markdown("Para comenzar con los análisis de datos, el primer paso es "
                    "cargar el conjunto de datos que vamos a estudiar. "
                    "Así que importa los datos y elige la función que quieras emplear. ")
        st.markdown("__Atención__:Si marca la casilla de limpiar datos, se borrarán tanto las filas duplicadas "
                    " como aquellas en las que falten valores, y se eliminarán los outliers.")

        st.sidebar.subheader('Opciones')

        clean_data = st.sidebar.checkbox(label='Limpiar datos')

        st.sidebar.subheader("Importe los datos")
        uploaded_file = st.sidebar.file_uploader(label="Importe aquí el archivo csv o excel.",
                                                accept_multiple_files=False,
                                                type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df, columns = load_dataframe(uploaded_file=uploaded_file, clean_data=clean_data)
            
            if link == 'AED':
                st.subheader("Bienvenido al entorno de Análisis Exploratorio de los Datos")
                st.markdown("Elija en el menú los parámetros a mostrar y comience con el análisis "
                            "de dataset.")
                analisis_exploratorio(df)

            if link == 'Visualizacion':
                st.subheader("Bienvenido al entorno de visualización")
                st.markdown("La visualización de datos dispone de varios temas a elegir, "
                            "así como múltiples tipos de gráficos que puede adaptar según "
                            "guste mediante parámetros diversos. De forma adicional, puede descargar el gráfico mostrado.")

                st.sidebar.subheader("Selección de tema")

                theme_selection = st.sidebar.selectbox(label="Seleccione el tema",
                                                    options=['plotly', 'plotly_white',
                                                                'ggplot2', 'seaborn', 
                                                                'simple_white', 'streamlit'])
                st.sidebar.subheader("Selección de gráfico")
                chart_type = st.sidebar.selectbox(label="Seleccione el tipo de gráfico.",
                                                options=['Grafico de dispersion', 'Grafico por sectores',
                                                            'Histograma', 'Grafico de lineas', 'Grafico de barras',
                                                            'Grafico de violin', 'Grafico de cajas',
                                                            'Mapa de calor'])

                control_graficos(chart_type=chart_type, df=df, dropdown_options=columns, template=theme_selection)
            
            if link == 'Informe':
                st.subheader("Información del dataset")
                st.markdown("A continuación se muestra un informe general sobre los datos importados, "
                            "tenga en cuenta que la creación de este informe no está recomendado para "
                            "Datasets de gran tamaño debido a la excesiva duración del proceso.")

                mostrar_informe(df=df) 

            if link == 'Machine Learning':
                st.subheader("Bienvenido al entorno de Machine Learning")
                st.markdown("Se encuntran disponibles tres tipos de aprendizaje automático con los siguientes algoritmos:")
                st.markdown("- **Clasificación**: regresión logística, SVC, naive bayes, árbol de decisión y random forest.")
                st.markdown("- **Regresión**: regresión lineal, random forest.")
                st.markdown("- **Agrupamiento**: k-means(prototipos), jerárquico y dbscan(densisad)")
                ml_selector(df)

                

    