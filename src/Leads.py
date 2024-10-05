import streamlit as st

from core.io import read_main_data

if __name__ == '__main__':
    st.set_page_config(
        page_title="Lead Tracking System",
        page_icon=":bar_chart:",
        layout="wide"
    )
    st.title("Lead Tracking System")

    df = read_main_data()

    if 'data' not in st.session_state:
        st.session_state['data'] = df

    st.write(df)
