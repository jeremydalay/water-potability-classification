from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
from ui_files.ui import (
    get_features, 
    get_random_state, 
    get_test_size,
    plot_corr_feat,
    plot_confusion
)

from model.NaiveBayes import (
    naive_bayes_model,
    pred
)

# Setup page
st.set_page_config(
    page_title = "Water Potability Classifier",
    layout = "wide"
)

data = pd.read_csv('https://github.com/jeremydalay/water-potability-classification/blob/main/dataset/water_potability_final.csv?raw=true',lineterminator='\n')
log = pd.DataFrame(columns = data.columns.values[:-1])

# Sidebar
def sidebar():
    # Configure dataset
    st.sidebar.title("Configure Dataset")
    test_size = get_test_size()
    random_state = get_random_state()
    
    # Set Features
    st.sidebar.title("Predict")
    input = get_features(data)
    
    if st.sidebar.button('Predict'):
        return [test_size, random_state, input]

    st.sidebar.markdown("---")
    return [test_size, random_state]


def body(user_input):
    # Title
    st.title("Water Potability Classifier")
    st.markdown("---")

    # Data Analysis
    st.subheader("A machine learning model that predicts water potability using Naive Bayes Algorithm.")
    st.write(data.describe())
    
    # Correlation Matrix in Content
    st.subheader('Correlation between features')
    fig_corr = plot_corr_feat(data.corr())
    st.plotly_chart(fig_corr)

    # Naive Bayes
    st.subheader('Gaussian Naive Bayes Model')
    NB, acc, cm = naive_bayes_model(data, user_input)
    fig_corr_acc = plot_confusion(cm)

    st.write('Confusion Matrix')
    st.plotly_chart(fig_corr_acc)

    col1, col2 = st.columns((1, 4))
    with col1:
        st.metric(
            "Accuracy", str(format(acc*100, '.2f'))+"%", 
            str(format(100-acc*100, '.2f'))+"%"
        )
    with col2:
        st.write('Simulation')
        
        if len(user_input) == 3:
            ans,log = pred(user_input[-1], NB,log)
            input = user_input[-1]
            st.text('Trial Logs')
            st.write(log)
            
        else:
            ans = ""
            input = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            st.text('Trial Logs')
            st.write(log)
            
        df = pd.DataFrame(
                np.array(input).reshape(-1, len(input)),
                columns=data.columns.drop("Potability")
        ) 
        df.index = ["Value"]
        st.write(df)
        st.subheader("Prediction: "+ans)
            
    
    # st.markdown("---")

    # Predict
    

    # Show Trial Logs
    #log = pd.read_csv('https://github.com/jeremydalay/water-potability-classification/blob/main/model/trial_logs.csv?raw=true',lineterminator='\n')
    #log = log.iloc[: , 1:]
    

if __name__ == "__main__":
    user_input = sidebar()
    body(user_input)