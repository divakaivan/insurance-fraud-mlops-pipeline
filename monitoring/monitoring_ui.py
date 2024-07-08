import streamlit as st
import streamlit.components.v1 as components

def full_dashaboard():
    """This function displays the full dashboard with the model and data dashboard"""
    st.title('Car Insurance Fraud Model and Data dashboard')
    path_to_html = "monitoring/evidently_reports/evidently_dashboard.html" 

    with open(path_to_html,'r', encoding='utf-8') as f: 
        html_data = f.read()
    components.html(html_data, height=5000, width=1000, scrolling=True)

def data_test_report():
    """This function displays the data tests dashboard"""
    st.title('Data tests dashboard')
    path_to_html = "monitoring/evidently_reports/data_stability.html" 

    with open(path_to_html,'r', encoding='utf-8') as f: 
        html_data = f.read()
    components.html(html_data, height=5000, width=1000, scrolling=True)

def shap_values():
    """This function displays the SHAP values dashboard"""
    st.image('monitoring/shap_lime_info/all_shap_values.png')
    st.image('monitoring/shap_lime_info/beeswarm.png')
    st.image('monitoring/shap_lime_info/fraud_shap_values.png')
    st.image('monitoring/shap_lime_info/not_fraud_shap_values.png')
    st.image('monitoring/shap_lime_info/mean_shap_values.png')

pg = st.navigation([
    st.Page(full_dashaboard, title="Model and Data dashboard", icon="📊"),
    st.Page(data_test_report, title="Data tests dashboard", icon="❗"),
    st.Page(shap_values, title='SHAP values', icon='📝')
])
pg.run()
