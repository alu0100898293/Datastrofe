from numpy.core.numeric import True_
import streamlit as st

from src.clasificacion import aplicar_clasificacion
from src.regresion import aplicar_regresion
from src.agrupamiento import aplicar_agrupamiento


def ml_selector(df):

    def set_features(df):
        features = st.multiselect('Seleccione la características, incluyendo la objetivo, a incluir en el modelo', list(df.columns) )
        return features

    def prepare_data_with_target(df, features):
        data = df[features]

         # Set target column
        target_options = data.columns
        chosen_target = st.sidebar.selectbox("Seleccione la característica objetivo", (target_options))

        X= data.loc[:, data.columns != chosen_target]
        X.columns = data.loc[:, data.columns != chosen_target].columns
        y = data[chosen_target]

        return X,y
    
    # Classifier type and algorithm selection 
    def set_model():
        type = st.sidebar.selectbox("Tipo de algoritmo", ("Clasificación", "Regresión", "Agrupamiento"))
        parameters={}
        chosen_classifier = None
        if type == "Regresión":
            st.sidebar.subheader("Regresión de Soporte Vectorial")
            parameters['maxIt'] = st.sidebar.slider('Máximo de iteraciones', 1, 100, 10)
            st.sidebar.subheader("Random Forest")
            parameters['trees'] = st.sidebar.slider('Número de árboles', 1, 1000, 10)
            parameters['maxDepth_rfr'] = st.sidebar.slider('Profundidad máxima del árbol', 0, 100, 10, key='rfr')

        elif type == "Clasificación":
            st.sidebar.subheader("Regresión Logística")
            parameters['maxIt'] = st.sidebar.slider('Máximo de iteraciones', 1, 100, 10)
            st.sidebar.subheader("Árbol de decisión")
            parameters['maxDepth_dt'] = st.sidebar.slider('Profundidad máxima del árbol', 0, 100, 10, key='dtc')
            st.sidebar.subheader("Random Forest")
            parameters['trees'] = st.sidebar.slider('Número de árboles', 1, 1000, 10)
            parameters['maxDepth_rfc'] = st.sidebar.slider('Profundidad máxima del árbol', 0, 100, 10, key='rfc')

        elif type == "Agrupamiento":
            st.sidebar.subheader("K-means") 
            parameters['nClusters'] = st.sidebar.slider('Número de clusters', 2, 10, 3, key='prototype')
            parameters['wcss'] = st.sidebar.checkbox(label='Mostrar curva wcss')

            st.sidebar.subheader("Jerárquico")
            parameters['nClusters_h'] = st.sidebar.slider('Número de clusters', 2, 10, 3, key='hierarchical') 
            parameters['dist'] = st.sidebar.selectbox(label="Método de distancia",
                                                    options=['single', 'complete',
                                                                'average',
                                                                'ward'])
            parameters['compDist'] = st.sidebar.checkbox(label='Comparar métodos de distancia')                                                    

            st.sidebar.subheader("DBScan")
            parameters['eps'] = st.sidebar.slider('Valor de epsilon', 0.0, 15.0, 1.0)
            parameters['minPts'] = st.sidebar.slider('Min. de puntos cercanos', 1, 20, 3)
            parameters['knnDist'] = st.sidebar.checkbox(label='Mostrar distancias entre k vecinos más cercanos')
        
        return type, chosen_classifier, parameters


    st.sidebar.subheader("Preparación de datos")
    seed=st.sidebar.slider('Seed',1,200)
    features=set_features(df)
    if len(features) > 1:
        type, chosen_classifier, parameters=set_model()

        if type == "Agrupamiento":
            X=df[features]
        else:
            X,y=prepare_data_with_target(df, features)
        model_btn = st.sidebar.button('Crear modelo')  

    if len(features) > 1 and model_btn:
        if type == "Clasificación":
            aplicar_clasificacion(X, y, seed, parameters)
            
        elif type == "Agrupamiento":
            aplicar_agrupamiento(X, seed, parameters)
        elif type == 'Regresión':
            aplicar_regresion(X, y, seed, parameters)
