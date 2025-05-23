"""
ONNX deployment component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.onnx_utils import (
    ONNXModelConverter, ONNXInferenceEngine, 
    train_and_convert_model, benchmark_inference
)
from src.data_processing import split_data


def create_onnx_deployment():
    """
    Create the ONNX deployment section of the application.
    """
    st.header("ONNX Model Deployment")
    
    # Check if data is loaded
    if st.session_state.df is None:
        st.warning("Please load data from the sidebar first.")
        return
    
    # Get data
    df = st.session_state.df
    
    # Model training and conversion section
    st.subheader("Train and Convert Model to ONNX")
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target column
        target_column = st.selectbox("Select target column", df.columns)
        
        # Select features
        feature_columns = st.multiselect(
            "Select feature columns",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column][:3]
        )
    
    with col2:
        # Select model type
        task_type = st.radio("Task type", ["Regression", "Classification"])
        
        if task_type == "Regression":
            model_options = ["linear_regression", "random_forest_regression"]
        else:
            model_options = ["logistic_regression", "random_forest_classification"]
        
        model_type = st.selectbox("Model type", model_options)
    
    # Train and convert button
    if st.button("Train and Convert to ONNX"):
        if not feature_columns:
            st.error("Please select at least one feature column.")
            return
        
        with st.spinner("Training model and converting to ONNX..."):
            try:
                # Prepare data
                X = df[feature_columns]
                y = df[target_column]
                
                # Split data
                splits = split_data(df, target_column, test_size=0.2, random_state=42)
                X_train, X_test = splits['X_train'], splits['X_test']
                y_train, y_test = splits['y_train'], splits['y_test']
                
                # Train and convert model
                model, onnx_path = train_and_convert_model(
                    X_train, y_train, model_type, "models/"
                )
                
                # Save model info to session state
                st.session_state.onnx_model_path = onnx_path
                st.session_state.onnx_feature_columns = feature_columns
                st.session_state.onnx_target_column = target_column
                st.session_state.onnx_test_data = (X_test, y_test)
                
                st.success(f"Model trained and converted to ONNX successfully!")
                st.success(f"ONNX model saved to: {onnx_path}")
                
            except Exception as e:
                st.error(f"Error during training/conversion: {str(e)}")
    
    # Model inference section
    if hasattr(st.session_state, 'onnx_model_path') and st.session_state.onnx_model_path:
        st.subheader("ONNX Model Inference")
        
        # Load existing ONNX model
        try:
            engine = ONNXInferenceEngine(st.session_state.onnx_model_path)
            
            # Display model info
            with st.expander("Model Information"):
                model_info = engine.get_model_info()
                st.json(model_info)
            
            # Single prediction
            st.markdown("### Single Prediction")
            
            # Create input fields
            input_values = {}
            cols = st.columns(len(st.session_state.onnx_feature_columns))
            
            for i, feature in enumerate(st.session_state.onnx_feature_columns):
                with cols[i]:
                    input_values[feature] = st.number_input(
                        f"{feature}", 
                        value=0.0, 
                        format="%.4f",
                        key=f"onnx_input_{feature}"
                    )
            
            if st.button("Predict with ONNX", key="single_prediction"):
                # Prepare input data
                input_array = np.array([[input_values[col] for col in st.session_state.onnx_feature_columns]], dtype=np.float32)
                
                # Make prediction
                start_time = time.perf_counter()
                prediction = engine.predict(input_array)
                inference_time = (time.perf_counter() - start_time) * 1000
                
                # Display results
                st.success(f"Prediction: {prediction[0]:.4f}")
                st.info(f"Inference time: {inference_time:.2f} ms")
            
            # Batch prediction
            st.markdown("### Batch Prediction")
            
            if hasattr(st.session_state, 'onnx_test_data'):
                X_test, y_test = st.session_state.onnx_test_data
                
                num_samples = st.slider("Number of test samples", 1, len(X_test), min(10, len(X_test)))
                
                if st.button("Run Batch Prediction", key="batch_prediction"):
                    # Prepare test data
                    test_data = X_test.iloc[:num_samples].values.astype(np.float32)
                    
                    # Make predictions
                    start_time = time.perf_counter()
                    predictions = engine.predict_batch(test_data)
                    total_time = (time.perf_counter() - start_time) * 1000
                    
                    # Display results
                    results_df = pd.DataFrame({
                        'Actual': y_test.iloc[:num_samples].values,
                        'Predicted': predictions.flatten()
                    })
                    
                    st.dataframe(results_df)
                    st.info(f"Total inference time for {num_samples} samples: {total_time:.2f} ms")
                    st.info(f"Average inference time per sample: {total_time/num_samples:.2f} ms")
                    
                    # Plot predictions vs actual
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.6)
                    ax.plot([results_df['Actual'].min(), results_df['Actual'].max()], 
                           [results_df['Actual'].min(), results_df['Actual'].max()], 'r--')
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('ONNX Model: Predicted vs Actual')
                    st.pyplot(fig)
            
            # Performance benchmark
            st.markdown("### Performance Benchmark")
            
            if st.button("Run Performance Benchmark"):
                if hasattr(st.session_state, 'onnx_test_data'):
                    X_test, _ = st.session_state.onnx_test_data
                    test_sample = X_test.iloc[:1].values.astype(np.float32)
                    
                    with st.spinner("Running benchmark..."):
                        benchmark_results = benchmark_inference(
                            st.session_state.onnx_model_path, 
                            test_sample, 
                            num_runs=100
                        )
                    
                    # Display benchmark results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Inference Time", f"{benchmark_results['mean_inference_time_ms']:.2f} ms")
                    with col2:
                        st.metric("Std Deviation", f"{benchmark_results['std_inference_time_ms']:.2f} ms")
                    with col3:
                        st.metric("Median Inference Time", f"{benchmark_results['median_inference_time_ms']:.2f} ms")
                    
                    # Plot benchmark results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    times = [benchmark_results[key] for key in ['min_inference_time_ms', 'median_inference_time_ms', 'mean_inference_time_ms', 'max_inference_time_ms']]
                    labels = ['Min', 'Median', 'Mean', 'Max']
                    ax.bar(labels, times)
                    ax.set_ylabel('Inference Time (ms)')
                    ax.set_title('Inference Time Statistics')
                    st.pyplot(fig)
                else:
                    st.warning("No test data available for benchmarking.")
            
        except Exception as e:
            st.error(f"Error loading ONNX model: {str(e)}")
    
    # Load existing ONNX model section
    st.subheader("Load Existing ONNX Model")
    
    # List available ONNX models
    models_dir = "models"
    if os.path.exists(models_dir):
        onnx_files = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
        
        if onnx_files:
            selected_onnx_model = st.selectbox("Select ONNX model", onnx_files)
            
            if st.button("Load Selected ONNX Model"):
                model_path = os.path.join(models_dir, selected_onnx_model)
                try:
                    engine = ONNXInferenceEngine(model_path)
                    st.session_state.onnx_model_path = model_path
                    st.success("ONNX model loaded successfully!")
                    
                    # Display model info
                    model_info = engine.get_model_info()
                    st.json(model_info)
                    
                except Exception as e:
                    st.error(f"Error loading ONNX model: {str(e)}")
        else:
            st.info("No ONNX models found in the models directory.")
    else:
        st.info("Models directory not found.")
