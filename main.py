import streamlit as st
import pandas as pd

from ui_files.ui import (
    get_features, 
    get_random_state, 
    get_test_size
)

# Setup page
st.set_page_config(
    page_title = "Water Potability Classifier",
    layout = "wide"
)

# Sidebar
def sidebar():
    data = pd.read_csv("dataset\water_potability.csv")

    # Configure dataset
    st.sidebar.title("Configure Dataset")
    test_size = get_test_size()
    random_state = get_random_state()
    
    # Set Features
    st.sidebar.title("Set Features")
    input = get_features(data)
    
    if st.sidebar.button('Predict'):
        return [test_size, random_state, input]

    st.sidebar.markdown("---")
    return [test_size, random_state]


def body(user_input):

    # Title
    st.title("Water Potability Classifier")
    st.markdown("---")

    # Prediction
    # Graph
    # Accuracy

    # Data Analysis
    


if __name__ == "__main__":
    user_input = sidebar()
    body(user_input)