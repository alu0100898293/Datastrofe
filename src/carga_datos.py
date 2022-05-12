import pandas as pd
import streamlit as st

@st.cache
def load_dataframe(uploaded_file, clean_data):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

    if clean_data:
        try:
            #Remove missing values
            df = df.dropna()
            #Remove duplicates
            df = df.drop_duplicates()
        except Exception as e:
            print(e)        

    columns = list(df.columns)
    columns.append(None)

    return df, columns