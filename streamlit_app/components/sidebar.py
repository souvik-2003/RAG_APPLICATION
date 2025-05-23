"""
Sidebar component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.data_processing import load_data, clean_data


def create_sidebar():
    """
    Create the sidebar for the application.
    
    Returns:
        Selected page
    """
    with st.sidebar:
        st.title("Navigation")
        
        # Page selection
        pages = ["Data Explorer", "Model Training", "Prediction"]
        selected_page = st.radio("Go to", pages)
        
        st.markdown("---")
        
        st.header("Data Loading")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            # Create data directories if they don't exist
            os.makedirs("data/raw", exist_ok=True)
            
            # Save uploaded file to disk temporarily
            with open(os.path.join("data/raw", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load data
            file_path = os.path.join("data/raw", uploaded_file.name)
            df = load_data(file_path)
            
            # Show data info
            st.write(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
            
            # Clean data option
            if st.checkbox("Clean data"):
                df = clean_data(df)
                st.write("Data cleaned!")
            
            # Save to session state
            st.session_state.df = df
        
        # Sample data option
        if st.button("Use Sample Data"):
            # Create sample data
            np.random.seed(42)
            data = {
                'feature1': np.random.normal(0, 1, 100),
                'feature2': np.random.normal(0, 1, 100),
                'feature3': np.random.normal(0, 1, 100),
                'target': np.random.normal(0, 1, 100)
            }
            df = pd.DataFrame(data)
            
            # Add a categorical feature
            df['category'] = pd.cut(df['feature1'], bins=3, labels=['Low', 'Medium', 'High'])
            
            # Save to session state
            st.session_state.df = df
            st.write("Sample data loaded!")
        
        st.markdown("---")
        
        # Show dataset info if loaded
        if st.session_state.df is not None:
            st.header("Dataset Info")
            st.write(f"Rows: {st.session_state.df.shape[0]}")
            st.write(f"Columns: {st.session_state.df.shape[1]}")
            
            # Show column types
            col_types = st.session_state.df.dtypes.value_counts()
            st.write("Column types:")
            for dtype, count in col_types.items():
                st.write(f"- {dtype}: {count}")
        
    return selected_page
