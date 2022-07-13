import streamlit as st
import pandas as pd

from ui_files.ui import (
    get_features, 
    get_random_state, 
    get_test_size
)

from model.NaiveBayes import naive_bayes_model

# Setup page
st.set_page_config(
    page_title = "Water Potability Classifier",
    layout = "wide"
)

# Sidebar
def sidebar():
    data = pd.read_csv("dataset\water_potability_final.csv")

    # Configure dataset
    st.sidebar.title("Configure Dataset")
    test_size = get_test_size()
    random_state = get_random_state()
    
    # Set Features
    st.sidebar.title("Set Features")
    input = get_features(data)
    
    if st.sidebar.button('Predict'):
        return [data, test_size, random_state, input]

    st.sidebar.markdown("---")
    return [data, test_size, random_state]


def body(user_input):

    # Title
    st.title("Water Potability Classifier")
    st.markdown("---")

    # Prediction
    if len(user_input) == 4:
        pred, acc = naive_bayes_model(user_input)
    


if __name__ == "__main__":
    user_input = sidebar()
    body(user_input)