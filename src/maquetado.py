import streamlit as st
from src.AED import analisis_exploratorio
from src.control_graficos import control_graficos
from src.carga_datos import load_dataframe, split
from src.informe import mostrar_informe
from src.machine_learning import control_clasificador, ml_selector
from streamlit.components.v1 import iframe
from src.machine_learning_v1 import ML
from pandas.errors import ParserError

def vistas(link):
    """
    Función de redirección entre páginas
    :param link: str, opción seleccionada en el radio button
    :return:
    """
    if link == 'Inicio':
        st.header('Bienvenidos al maravilloso mundo del anáilisis de datos')
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

            if link == 'Machine Learning':
                st.subheader("Bienvenido al entorno de Machine Learning")
                ml_selector(df)
                #controller = ML()
                #try:
                #    controller.data = df

                #    if controller.data is not None:
                #        split_data = st.sidebar.slider('Randomly reduce data size %', 1, 100, 10 )
                #        train_test = st.sidebar.slider('Train-test split %', 1, 99, 66 )
                #    controller.set_features()
                #    if len(controller.features) > 1:
                #        controller.prepare_data(split_data, train_test)
                #        controller.set_classifier_properties()
                #        predict_btn = st.sidebar.button('Predict')  
                #except (AttributeError, ParserError, KeyError) as e:
                #    st.markdown('<span style="color:blue">WRONG FILE TYPE</span>', unsafe_allow_html=True)  


                #if controller.data is not None and len(controller.features) > 1:
                #    if predict_btn:
                #        st.sidebar.text("Progress:")
                #        my_bar = st.sidebar.progress(0)
                #        predictions, predictions_train, result, result_train = controller.predict(predict_btn)
                #        for percent_complete in range(100):
                #            my_bar.progress(percent_complete + 1)
                #        
                #        controller.get_metrics()        
                #        controller.plot_result()
                #        controller.print_table()

                

    