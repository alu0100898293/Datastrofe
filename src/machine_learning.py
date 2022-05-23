from numpy.core.numeric import True_
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plot

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
        type = st.sidebar.selectbox("Tipo de algoritmo", ("Clasificación", "Regresión", "Clustering"))
        parameter=None
        if type == "Regresión":
            chosen_classifier = st.sidebar.selectbox("Escoja un clasificador", ('Random Forest', 'Linear Regression')) 
            if chosen_classifier == 'Random Forest': 
                parameter = st.sidebar.slider('Número de aŕboles', 1, 1000, 1)
        elif type == "Clasificación":
            chosen_classifier = st.sidebar.selectbox("Escoja un clasificador", ('Logistic Regression', 'Naive Bayes')) 
            if chosen_classifier == 'Logistic Regression': 
                parameter = st.sidebar.slider('Máximo de iteraciones', 1, 100, 10)
        
        elif type == "Clustering":
            pass
        
        return type, chosen_classifier, parameter

    def predict_classification(X, y, seed, type, chosen_classifier, parameter): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        predictions = None
        predictions_train = None

        if type == "Regresión":    
            if chosen_classifier == 'Random Forest':
                alg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=parameter)
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
                st.markdown('#### Ecuación obtenida: ' + str(alg.coef_) +
            'x + ' + str(alg.intercept_))

        elif type == "Clasificación":
            if chosen_classifier == 'Logistic Regression':
                alg = LogisticRegression(max_iter=parameter)
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

    def predict_clustering(X, y, seed, chosen_classifier):
        st.subheader("En progreso")

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
    
    def plot_metrics(model, X_test, X_train, result, result_train, type):
        col1, col2 = st.columns(2)
        if type == 'Clasificación':
            display = ConfusionMatrixDisplay.from_estimator(
                model,
                X_train,
                result_train.Actual_Train,
                cmap=plot.cm.Blues,
                normalize=None,
            )
            display.ax_.set_title("Matriz de confusión en entrenamiento")
            col1.pyplot()

            display = ConfusionMatrixDisplay.from_estimator(
                model,
                X_test,
                result.Actual,
                cmap=plot.cm.Blues,
                normalize=None,
            )
            display.ax_.set_title("Matriz de confusión en test")
            col2.pyplot()

    

    st.sidebar.subheader("Preparación de datos")
    seed=st.sidebar.slider('Seed',1,200)
    features=set_features(df)
    if len(features) > 1:
        type, chosen_classifier, parameter=set_model()

        if type == "Clustering":
            X=df[features]
        else:
            X,y=prepare_data_with_target(df, features)
        model_btn = st.sidebar.button('Crear modelo')  

    if len(features) > 1 and model_btn:
        model, X_train, X_test, result, result_train = predict_classification(X, y, seed, type, chosen_classifier, parameter)
        get_metrics(type, result, result_train)
        plot_metrics(model, X_test, X_train, result, result_train, type)