import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def aplicar_clasificacion(X, y, seed, parameters):
    """
    Función que ejecuta un pipeline con distintos tipos de algoritmos de calsificación
    :param X: datos de variables predictoras
    :param y: datos de variable de clase
    :param seed: semilla para añadir variación en la división de los datos
    :param parameters: dict, diversos valores para los algoritmos de clasificación
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

    def plot_matrix(estimator, X, y, type, st):
        display = ConfusionMatrixDisplay.from_estimator(
            estimator,
            X,
            y,
            cmap=plt.cm.Blues,
            normalize=None,
        )
        display.ax_.set_title("Matriz de confusión en " + type)
        st.pyplot()
    
    def plot_tree_from_pipeline(model, columns, target):
        fig = plt.figure(figsize=(15, 10))
        plot_tree(model, 
                feature_names=columns, class_names=target,
                filled=True, impurity=True, 
                rounded=True, max_depth=3,
                fontsize=9)
        st.pyplot()

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
        ("clf", LogisticRegression())
    ])

    clfs = []
    clfs.append(LogisticRegression(max_iter=parameters['maxIt']))
    clfs.append(LinearSVC())
    clfs.append(GaussianNB())
    clfs.append(DecisionTreeClassifier(max_depth=parameters['maxDepth_dt']))
    clfs.append(RandomForestClassifier(n_estimators=parameters['trees'], max_depth=parameters['maxDepth_rfc']))


    for classifier in clfs:
        st.markdown("___")
        st.markdown("#### "+classifier.__class__.__name__)

        pipeline.set_params(clf = classifier)

        pipeline.fit(X_train, y_train)

        col1, col2 = st.columns(2)
        col1.markdown("**Precisión en entrenamiento**: "+ str(round(pipeline.score(X_train, y_train), 3)))
        col2.markdown("**Precisión en test**: "+ str(round(pipeline.score(X_test, y_test), 3)))

        plot_matrix(pipeline, X_train, y_train, 'entrenamiento', col1)

        plot_matrix(pipeline, X_test, y_test, 'test', col2)

        if(classifier.__class__.__name__ == "DecisionTreeClassifier"):
            st.markdown("**Profundidad del árbol**: "+ str(pipeline['clf'].tree_.max_depth))
            plot_tree_from_pipeline(pipeline['clf'], X.columns, list(set(y)))


        if(classifier.__class__.__name__ == "RandomForestClassifier"):
            n_tree = st.number_input('Seleccione el árbol a mostrar', min_value=1, max_value=len(pipeline['clf'].estimators_), value=1, step=1)
            plot_tree_from_pipeline(pipeline['clf'].estimators_[n_tree-1], X.columns, list(set(y)))