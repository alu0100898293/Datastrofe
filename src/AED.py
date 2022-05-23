import streamlit as st


def analisis_exploratorio(df):

    try:
        st.subheader("Primeras filas del dataset")
        st.sidebar.subheader("Datos a mostrar del Dataset")
        number_of_rows = st.sidebar.number_input(label='Seleccione el número de filas', min_value=2)

        st.dataframe(df.head(number_of_rows))
    except Exception as e:
        print(e)
        
    st.sidebar.subheader("Parámetros de análisis")

    if st.sidebar.checkbox("Mostrar dimensionalidad (shape)"):
        st.subheader("Dimensionalidad")
        st.write(df.shape)

    if st.sidebar.checkbox("Mostrar columnas"):
        st.subheader("Columnas")
        st.write(df.columns)

    if st.sidebar.checkbox("Mostrar resumen"):
        st.subheader("Resumen")
        st.write(df.describe().T)

    if st.sidebar.checkbox('Mostrar valores nulos'):
        st.subheader("Valores nulos")
        st.write(df.isnull().sum())

    if st.sidebar.checkbox("Mostrar tipos de datos"):
        st.subheader("Tipos de datos")
        st.write(df.dtypes)

    if st.sidebar.checkbox('Mostrar correlación de los datos'):
        st.subheader("Correlación de los datos")
        st.write(df.corr())