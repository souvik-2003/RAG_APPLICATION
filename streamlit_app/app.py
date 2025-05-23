"""
Main Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from streamlit_app.components.sidebar import create_sidebar
from streamlit_app.components.data_explorer import create_data_explorer
from streamlit_app.components.model_training import create_model_training
from streamlit_app.components.prediction import create_prediction

# Set page configuration
st.set_page_config(
    page_title="Data Science App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create session state if it doesn't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'train_test_split' not in st.session_state:
    st.session_state.train_test_split = None


def main():
    """Main application function."""
    st.title("Data Science Application")
    st.markdown("""
    This application allows you to:
    - Load and explore data
    - Preprocess and visualize data
    - Train machine learning models
    - Make predictions
    """)
    
    # Create sidebar
    selected_page = create_sidebar()
    
    # Show selected page
    if selected_page == "Data Explorer":
        create_data_explorer()
    elif selected_page == "Model Training":
        create_model_training()
    elif selected_page == "Prediction":
        create_prediction()
        

if __name__ == "__main__":
    main()
