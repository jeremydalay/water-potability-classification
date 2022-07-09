import streamlit as st
import pandas as pd

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
    # st.sidebar.markdown("---")

    test_size = st.sidebar.slider(
        "Set Test Size",
        min_value = 0.10,
        max_value = 0.50,
        step = 0.10,
        value = 0.30
    )
    random_state = st.sidebar.number_input(
        "Set Random State",
        min_value = 0,
        max_value = 42,
        value = 40
    )
    
    # Set Features
    st.sidebar.title("Set Features")
    features = list(data.columns.drop(data.columns[-1]))
    input = []
    
    for feat in features:
        with st.sidebar.expander(feat, False):
            input.append(
                st.number_input("Set " + feat)
            )
    
    if st.sidebar.button('Predict'):
        pass

    st.sidebar.markdown("---")


def body():

    # Title
    st.title("Water Potability Classifier")
    st.markdown("---")


if __name__ == "__main__":
    sidebar()
    body()