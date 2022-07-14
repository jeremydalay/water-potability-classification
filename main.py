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

data = pd.read_csv(r"dataset\water_potability_final.csv")

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
    st.subheader('Data Analysis')
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
            ans = pred(user_input[-1], NB)
            input = user_input[-1]
            
        else:
            ans = ""
            input = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
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
    log = pd.read_csv(r"model\trial_logs.csv")
    log = log.iloc[: , 1:]
    st.text('Trial Logs')
    st.write(log)

if __name__ == "__main__":
    user_input = sidebar()
    body(user_input)