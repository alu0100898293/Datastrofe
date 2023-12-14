import streamlit as st


def analisis_exploratorio(df):

    try:
        st.subheader("Primeras filas del dataset")
        st.sidebar.subheader("Datos a mostrar del Dataset")
        number_of_rows = st.sidebar.number_input(label='Seleccione el número de filas', min_value=2)

        st.dataframe(df.head(number_of_rows))
    except Exception as e:
        print(e)
        
    st.subheader("Parámetros de análisis")

    dim_expander = st.expander(label='Dimensionalidad (shape)')
    with dim_expander:
        st.write(df.shape)

    col_expander = st.expander(label='Columnas')
    with col_expander:
        st.write(df.columns)

    sum_expander = st.expander(label='Resumen')
    with sum_expander:
        st.write(df.describe().T)

    null_expander = st.expander(label='Valores nulos')
    with null_expander:
        st.write(df.isnull().sum())

    type_expander = st.expander(label='Tipos de datos')
    with type_expander:
        st.write(df.dtypes.astype(str))

    corr_expander = st.expander(label='Correlación de los datos')
    with corr_expander:
        try:
            st.write(df.corr())
        except Exception as e:
            st.error("Se produjo el siguiente error al calcular la matriz de correlación:")    
            st.error(e)