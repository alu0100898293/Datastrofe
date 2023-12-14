import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.linear_model import BayesianRidge, LinearRegression

def aplicar_regresion(X, y, seed, parameters):
    """
    Función que ejecuta un pipeline con distintos tipos de algoritmos de regresion
    :param X: datos de variables predictoras
    :param y: datos de variable de clase
    :param seed: semilla para añadir variación en la división de los datos
    :param parameters: dict, diversos valores para los algoritmos de regresion
    :return:
    """
    def get_columns_types(df):
        numeric_features = []
        categorical_features = []

        numerics = df._get_numeric_data().columns
        
        for column in numerics:
            numeric_features.append(column)

        categoricals = list(set(df.columns) - set(numerics))

        for column in categoricals:
            categorical_features.append(column)

        return numeric_features, categorical_features
    
    def plot_tree_from_pipeline(model, columns, target):
        fig = plt.figure(figsize=(15, 10))
        plot_tree(model, 
                feature_names=columns, class_names=target,
                filled=True, impurity=True, 
                rounded=True, max_depth=3,
                fontsize=9)
        st.pyplot()

    if parameters['maxDepth_rfr'] == 0:
        parameters['maxDepth_rfr'] = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    numeric_features, categorical_features = get_columns_types(X_train)

    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())
        ]
    )

    categorical_tranformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_tranformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rgr", None)
    ])

    rgrs = []
    rgrs.append(LinearRegression())
    rgrs.append(LinearSVR(max_iter=parameters['maxIt']))
    rgrs.append(BayesianRidge())
    rgrs.append(DecisionTreeRegressor(max_depth=4))
    rgrs.append(RandomForestRegressor(random_state=seed, n_estimators=parameters['trees'], max_depth=parameters['maxDepth_rfr']))


    for regressor in rgrs:
        st.markdown("___")
        st.markdown("#### "+regressor.__class__.__name__)

        try:
            pipeline.set_params(rgr = regressor)

            pipeline.fit(X_train, y_train)
            pred_train = pipeline.predict(X_train)
            pred_test = pipeline.predict(X_test)

            col1, col2 = st.columns(2)
            col1.markdown("**Precisión en entrenamiento**: "+ str(round(pipeline.score(X_train, y_train), 3)))
            col2.markdown("**Precisión en test**: "+ str(round(pipeline.score(X_test, y_test), 3)))

            error_metrics = {}
            error_metrics['MSE_test'] = mean_squared_error(y_train, pred_train)
            error_metrics['MSE_train'] = mean_squared_error(y_test, pred_test)
            col1.markdown("**MSE en entrenamiento**: "+ str(round(error_metrics['MSE_train'], 3)))
            col2.markdown("**MSE en test**: "+ str(round(error_metrics['MSE_test'], 3)))
            
            if(regressor.__class__.__name__ == "LinearRegression"):
                st.markdown('**Ecuación obtenida**: ' + str(pipeline['rgr'].coef_) +
                    'x + ' + str(pipeline['rgr'].intercept_))
            
            fig, ax = plt.subplots()
            ax.scatter(pred_test, y_test, edgecolors=(0, 0, 1))
            ax.plot([y_test.min(), y_test.max()], [pred_test.min(), pred_test.max()], 'r--', lw=3)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot()

            if(regressor.__class__.__name__ == "DecisionTreeRegressor"):
                plot_tree_from_pipeline(pipeline['rgr'], X_train.columns, list(set(y)))
                
        except Exception as e:
            st.error("Se produjo el siguiente error al crear el modelo:")
            st.error(e)