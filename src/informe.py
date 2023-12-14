from ydata_profiling import ProfileReport
import streamlit as st

from streamlit_pandas_profiling import st_profile_report

def mostrar_informe(df):
    try:
        report = ProfileReport(df)
        st_profile_report(report)
    except Exception as e:
        st.error("Se produjo el siguiente error al calcular la matriz de correlación:")    
        st.error(e)