import streamlit as st

import pandas as pd

file_formats_dataframe = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

files_formats_documents = ["pdf", "docx", "doc"]


def clear_submit():
    st.session_state["submit"] = False
