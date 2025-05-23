"""
Prediction component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.models import load_model


def create_prediction():
    """
    Create the prediction section of the application.
    """
    st.header("Prediction")
    
    # Check if model is loaded or trained
    if st.session_state.model is None:
        st.warning("Please train a model first or load a saved model.")
        
        # Option to load a saved model
        st.subheader("Load Saved Model")
        
        # List available models
        models_dir = "models"
        if os.path.exists(models_dir) and os.listdir(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if model_files:
                selected_model = st.selectbox("Select a model", model_files)
                
                if st.button("Load Model"):
                    model_path = os.path.join(models_dir, selected_model)
                    try:
                        model = load_model(model_path)
                        st.session_state.model = model
                        st.success("Model loaded successfully!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
            else:
                st.info("No saved models found.")
        else:
            st.info("No saved models found.")
        
        return
    
    # Get model
    model = st.session_state.model
    
    # Display model info
    st.subheader("Model Information")
    st.write(f"Model type: {type(model).__name__}")
    
    # Input method selection
    input_method = st.radio("Select input method", ["Manual Input", "File Upload"])
    
    if input_method == "Manual Input":
        # If we have feature names in session state
        if 'feature_names' in st.session_state:
            features = st.session_state.feature_names
            
            # Create input fields for each feature
            st.subheader("Enter Feature Values")
            
            input_values = {}
            for feature in features:
                # Check if feature is categorical
                if (st.session_state.df is not None and 
                    feature in st.session_state.df.columns and 
                    st.session_state.df[feature].dtype == 'object'):
                    # Create a dropdown for categorical features
                    unique_values = st.session_state.df[feature].unique().tolist()
                    input_values[feature] = st.selectbox(f"{feature}", unique_values)
                else:
                    # Create a number input for numerical features
                    input_values[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
            
            # Make prediction
            if st.button("Make Prediction"):
                # Convert input to DataFrame
                input_df = pd.DataFrame([input_values])
                
                # Make prediction
                prediction = model.predict(input_df)
                
                # Display prediction
                st.subheader("Prediction")
                st.success(f"Predicted value: {prediction[0]:.4f}" if isinstance(prediction[0], (int, float)) else f"Predicted class: {prediction[0]}")
        else:
            st.warning("Feature information not available. Please train a model first.")
    
    else:  # File Upload
        st.subheader("Upload Prediction Data")
        
        # Upload file
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith(".csv"):
                    input_df = pd.read_csv(uploaded_file)
                else:
                    input_df = pd.read_excel(uploaded_file)
                
                # Display data
                st.write("Preview of uploaded data:")
                st.dataframe(input_df.head())
                
                # Check if all required features are present
                if 'feature_names' in st.session_state:
                    missing_features = [f for f in st.session_state.feature_names if f not in input_df.columns]
                    
                    if missing_features:
                        st.error(f"Missing required features: {', '.join(missing_features)}")
                    else:
                        # Make prediction
                        if st.button("Make Predictions"):
                            # Extract features
                            X = input_df[st.session_state.feature_names]
                            
                            # Make predictions
                            predictions = model.predict(X)
                            
                            # Add predictions to DataFrame
                            input_df['Prediction'] = predictions
                            
                            # Display results
                            st.subheader("Predictions")
                            st.dataframe(input_df)
                            
                            # Option to download results
                            csv = input_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )
                else:
                    st.warning("Feature information not available. Please train a model first.")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
