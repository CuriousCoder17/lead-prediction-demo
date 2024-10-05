import streamlit as st
import plotly_express as px

from core.io import read_main_data

st.title("Lead Insights")
data = read_main_data()

with st.expander("Overall Demographics"):
    st.bar_chart(
        data['Location'].value_counts(), x_label="City", y_label="# Leads"
    )

    st.bar_chart(
        data['DeviceType'].value_counts(),
        x_label="Device Type", y_label="# Leads", color='#ffaa00'
    )

    fig = px.pie(
        data.groupby('Conversion (Target)')['LeadID'].count().reset_index(),
        values='LeadID',
        names='Conversion (Target)',
        title="% Conversions",
    )
    st.plotly_chart(fig, theme=None)
