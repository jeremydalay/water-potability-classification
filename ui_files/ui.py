import streamlit as st
import pandas as pd

def get_test_size():
    return st.sidebar.slider(
        "Set Test Size",
        min_value = 0.10,
        max_value = 0.50,
        step = 0.10,
        value = 0.30
    )

def get_random_state():
    return st.sidebar.number_input(
        "Set Random State",
        min_value = 0,
        max_value = 42,
        value = 40
    )

def get_features(data):
    attributes = list(data.columns.drop(data.columns[-1]))
    input = []
    
    for col in attributes:
        with st.sidebar.expander(col, False):
            input.append(
                st.number_input("Set " + col)
            )
    
    return input