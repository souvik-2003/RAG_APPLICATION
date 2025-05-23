"""
Model Training component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.data_processing import split_data
from src.models import (
    train_linear_regression, train_random_forest, 
    evaluate_regression_model, evaluate_classification_model,
    save_model
)
from src.visualization import plot_feature_importance, plot_prediction_vs_actual


def create_model_training():
    """
    Create the model training section of the application.
    """
    st.header("Model Training")
    
    # Check if data is loaded
    if st.session_state.df is None:
        st.warning("Please load data from the sidebar first.")
        return
    
    # Get data
    df = st.session_state.df
    
    # Model configuration
    st.subheader("Model Configuration")
    
    # Select task type
    task_type = st.radio(
        "Select task type",
        ["Regression", "Classification"]
    )
    
    # Select target column
    target_column = st.selectbox("Select target column", df.columns)
    
    # Select features
    feature_columns = st.multiselect(
        "Select feature columns",
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column]
    )
    
    if not feature_columns:
        st.warning("Please select at least one feature column.")
        return
    
    # Select model type
    model_type = st.selectbox(
        "Select model type",
        ["Linear Regression", "Random Forest"] if task_type == "Regression" else ["Logistic Regression", "Random Forest"]
    )
    
    # Model hyperparameters
    with st.expander("Model Hyperparameters"):
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of estimators", 10, 500, 100, 10)
            max_depth = st.slider("Maximum depth", 1, 50, 10, 1)
    
    # Train/test split
    with st.expander("Train/Test Split"):
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 1000, 42, 1)
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Prepare data
            X = df[feature_columns]
            y = df[target_column]
            
            # Split data
            splits = split_data(df, target_column, test_size=test_size, random_state=random_state)
            X_train, X_test = splits['X_train'], splits['X_test']
            y_train, y_test = splits['y_train'], splits['y_test']
            
            # Save split to session state
            st.session_state.train_test_split = splits
            
            # Train model
            if model_type == "Linear Regression":
                model = train_linear_regression(X_train, y_train)
                st.session_state.model_type = "regression"
            elif model_type == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=random_state)
                model.fit(X_train, y_train)
                st.session_state.model_type = "classification"
            elif model_type == "Random Forest":
                if task_type == "Regression":
                    model = train_random_forest(
                        X_train, y_train, 
                        is_classification=False,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                    st.session_state.model_type = "regression"
                else:
                    model = train_random_forest(
                        X_train, y_train, 
                        is_classification=True,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                    st.session_state.model_type = "classification"
            
            # Save model to session state
            st.session_state.model = model
            
            # Save feature names
            st.session_state.feature_names = feature_columns
            
            # Display success message
            st.success("Model trained successfully!")
    
    # Model evaluation
    if st.session_state.model is not None:
        st.subheader("Model Evaluation")
        
        # Get model and split data
        model = st.session_state.model
        splits = st.session_state.train_test_split
        
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        # Evaluate model
        if st.session_state.model_type == "regression":
            metrics = evaluate_regression_model(model, X_test, y_test)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{metrics['rmse']:.4f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            # Plot predictions vs actual
            st.subheader("Predictions vs Actual")
            y_pred = model.predict(X_test)
            fig = plot_prediction_vs_actual(y_test.values, y_pred)
            st.pyplot(fig)
        else:
            metrics = evaluate_classification_model(model, X_test, y_test)
            
            # Display metrics
            metric_cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                with metric_cols[i]:
                    st.metric(metric_name.capitalize(), f"{metric_value:.4f}")
            
            # Plot confusion matrix
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)
        
        # Feature importance for Random Forest
        if model_type == "Random Forest":
            st.subheader("Feature Importance")
            fig = plot_feature_importance(model, st.session_state.feature_names)
            st.pyplot(fig)
        
        # Save model option
        st.subheader("Save Model")
        model_name = st.text_input("Model filename", "model")
        if st.button("Save Model"):
            # Create directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save model
            model_path = f"models/{model_name}.pkl"
            save_model(model, model_path)
            st.success(f"Model saved to {model_path}")
