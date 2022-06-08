from numpy.core.numeric import True_
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from src.clasificacion import aplicar_clasificacion
from src.regresion import aplicar_regresion
from src.agrupamiento import aplicar_agrupamiento

##sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree

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
            parameters['nClusters'] = st.sidebar.slider('Número de clusters', 2, 10, 3, key='hierarchical')
            parameters['wcss'] = st.sidebar.checkbox(label='Mostrar curva wcss')

            st.sidebar.subheader("Jerárquico")
            parameters['nClusters_h'] = st.sidebar.slider('Número de clusters', 2, 10, 3) 
            parameters['dist'] = st.sidebar.selectbox(label="Método de distancia",
                                                    options=['single', 'complete',
                                                                'average',
                                                                'ward'])
            parameters['compDist'] = st.sidebar.checkbox(label='Comparar métodos de distancia')                                                    

            st.sidebar.subheader("DBScan")
            parameters['eps'] = st.sidebar.slider('Valor de epsilon', 0.0, 30.0, 1.0)
            parameters['minPts'] = st.sidebar.slider('Min. de puntos cercanos', 1, 20, 3)
            parameters['knnDist'] = st.sidebar.checkbox(label='Mostrar distancias entre k vecinos más cercanos')


        
        return type, chosen_classifier, parameters

    def predict_classification(X, y, seed, type, chosen_classifier, parameters): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        predictions = None
        predictions_train = None

        if type == "Regresión":    
            if chosen_classifier == 'Random Forest':
                alg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=parameters)
                model = alg.fit(X_train, y_train)
                predictions = alg.predict(X_test)
                predictions_train = alg.predict(X_train)
                predictions = predictions
                
            
            elif chosen_classifier=='Linear Regression':
                alg = LinearRegression()
                model = alg.fit(X_train, y_train)
                predictions = alg.predict(X_test)
                predictions_train = alg.predict(X_train)
                predictions = predictions

        elif type == "Clasificación":
            if chosen_classifier == 'Logistic Regression':
                alg = LogisticRegression(max_iter=parameters)
                model = alg.fit(X_train, y_train)
                predictions = alg.predict(X_test)
                predictions_train = alg.predict(X_train)
                predictions = predictions
        
            elif chosen_classifier=='Naive Bayes':
                alg = GaussianNB()
                model = alg.fit(X_train, y_train)
                predictions = alg.predict(X_test)
                predictions_train = alg.predict(X_train)
                predictions = predictions
        
        result = pd.DataFrame(columns=['Actual', 'Actual_Train', 'Prediction', 'Prediction_Train'])
        result_train = pd.DataFrame(columns=['Actual_Train', 'Prediction_Train'])
        result['Actual'] = y_test
        result_train['Actual_Train'] = y_train
        result['Prediction'] = predictions
        result_train['Prediction_Train'] = predictions_train
        result.sort_index()
        result = result
        result_train = result_train

        return model, X_train, X_test, result, result_train

    def predict_Agrupamiento(X, seed, chosen_classifier, parameters):
        #https://github.com/thuwarakeshm/Streamlit-Intro/blob/main/quickstart.py
        if chosen_classifier == 'K-means':
            alg = KMeans(parameters, random_state=seed)
            model = alg.fit(X)
            wcss = []
            for i in range(1, 10):
                kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500, n_init = 10, random_state = seed)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)

        return model, wcss

    def get_metrics(type, result, result_train):
        st.subheader("Métricas")
        error_metrics = {}
        if type == 'Regresión':
            error_metrics['MSE_test'] = mean_squared_error(result.Actual, result.Prediction)
            error_metrics['MSE_train'] = mean_squared_error(result_train.Actual_Train, result_train.Prediction_Train)
            return st.markdown('#### MSE de Entrenamiento: ' + str(round(error_metrics['MSE_train'], 3)) + 
            ' -- MSE de Test: ' + str(round(error_metrics['MSE_test'], 3)))

        elif type == 'Clasificación':
            error_metrics['Accuracy_test'] = accuracy_score(result.Actual, result.Prediction)
            error_metrics['Accuracy_train'] = accuracy_score(result_train.Actual_Train, result_train.Prediction_Train)
            return st.markdown('#### Precisión de Entrenamiento: ' + str(round(error_metrics['Accuracy_train'], 3)) +
            ' -- Precisión de Test: ' + str(round(error_metrics['Accuracy_test'], 3)))
    
    def plot_confusion_matrix(model, X_test, X_train, result, result_train):
        col1, col2 = st.columns(2)
        display = ConfusionMatrixDisplay.from_estimator(
            model,
            X_train,
            result_train.Actual_Train,
            cmap=plt.cm.Blues,
            normalize=None,
        )
        display.ax_.set_title("Matriz de confusión en entrenamiento")
        col1.pyplot()

        display = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            result.Actual,
            cmap=plt.cm.Blues,
            normalize=None,
        )
        display.ax_.set_title("Matriz de confusión en test")
        col2.pyplot()

    def plot_regresion(model, df, chosen_classifier):
        
        if chosen_classifier == 'Random Forest':
            tree = st.number_input('Select tree', min_value=1, max_value=len(model.estimators_), value=1, step=1)
            fig = plt.figure(figsize=(15, 10))
            plot_tree(model.estimators_[tree-1], 
                    feature_names=df.columns,
                    filled=True, impurity=True, 
                    rounded=True)
            st.pyplot()

        elif chosen_classifier=='Linear Regression':
            st.markdown('#### Ecuación obtenida: ' + str(model.coef_) +
            'x + ' + str(model.intercept_))
            if len(df.columns) == 2:
                plot = px.scatter(data_frame=df, x=df.columns[0], y=df.columns[1], 
                    trendline="ols", trendline_color_override='darkblue')
                st.subheader("Recta de regresión")
                st.plotly_chart(plot)

    def plot_cluster(model, X, wcss):
        st.subheader("Centroides")
        centroids = pd.DataFrame(model.cluster_centers_, columns = X.columns)
        st.table(centroids)

        fig = go.Figure(data = go.Scatter(x = [1,2,3,4,5,6,7,8,9,10], y = wcss))
        fig.update_layout(title='WCSS vs. Número de clusters',
                        xaxis_title='Clusters',
                        yaxis_title='WCSS')
        st.plotly_chart(fig)

        if len(X.columns) == 2:
            X['Cluster'] = model.labels_
            fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color='Cluster')
            st.plotly_chart(fig)

        elif len(X.columns) == 3:
            X['Cluster'] = model.labels_
            fig = px.scatter_3d(X, x=X.columns[0], y=X.columns[1], z=X.columns[2],
              color='Cluster', opacity = 0.8, size=X.columns[0], size_max=30)
            st.plotly_chart(fig)

    

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
