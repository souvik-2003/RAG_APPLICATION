"""
Data Explorer component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.visualization import plot_distribution, plot_correlation_matrix


def create_data_explorer():
    """
    Create the data explorer section of the application.
    """
    st.header("Data Explorer")
    
    # Check if data is loaded
    if st.session_state.df is None:
        st.warning("Please load data from the sidebar first.")
        return
    
    # Get data
    df = st.session_state.df
    
    # Show data options
    st.subheader("Data Overview")
    
    # Select rows to display
    num_rows = st.slider("Number of rows to display", 5, 100, 10)
    
    # Display data
    st.dataframe(df.head(num_rows))
    
    # Summary statistics
    with st.expander("Summary Statistics"):
        st.write(df.describe())
    
    # Missing values
    with st.expander("Missing Values"):
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing = missing[missing > 0]
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing Values': missing,
                'Percentage': missing_pct
            })
            st.write(missing_df)
            
            # Plot missing values
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df.sort_values('Missing Values', ascending=False).plot(
                kind='bar', ax=ax, ylabel='Count / Percentage'
            )
            st.pyplot(fig)
        else:
            st.write("No missing values found!")
    
    # Data visualization
    st.subheader("Data Visualization")
    
    # Variable distribution
    st.markdown("### Variable Distribution")
    
    # Select column for distribution
    column = st.selectbox("Select column for distribution plot", df.columns)
    
    # Plot distribution
    fig = plot_distribution(df[column], f'Distribution of {column}')
    st.pyplot(fig)
    
    # Correlation matrix
    st.markdown("### Correlation Matrix")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Select columns for correlation
        selected_cols = st.multiselect(
            "Select columns for correlation matrix", 
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        if selected_cols:
            # Plot correlation matrix
            fig = plot_correlation_matrix(df[selected_cols])
            st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation analysis.")
