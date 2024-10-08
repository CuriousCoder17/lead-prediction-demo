import streamlit as st
import plotly_express as px

from core.io import read_main_data

st.title("Lead Insights")
data = read_main_data()

with st.expander("Overall Demographics", expanded=True):
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

with st.expander("Conversion Stats", expanded=True):
    conv_loc = data \
        .groupby('Location')['Conversion (Target)'].sum().reset_index()

    fig = px.pie(
        conv_loc, values='Conversion (Target)', names='Location', hole=.3,
        title="Conversion percentage by Location"
    )
    st.plotly_chart(fig, theme=None)

    status_loc = data.groupby('LeadStatus').agg(
        {'Conversion (Target)': 'sum'}
    ).reset_index()

    st.bar_chart(
        status_loc,
        x='LeadStatus',
        y='Conversion (Target)',
        x_label="LeadStatus", y_label="# Conversions", color='#ffaa00'
    )
