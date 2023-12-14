from ydata_profiling import ProfileReport

from streamlit_pandas_profiling import st_profile_report

def mostrar_informe(df):
    report = ProfileReport(df)
    st_profile_report(report)