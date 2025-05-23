"""
ONNX utilities for model conversion and deployment.
"""
import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Union, List, Tuple, Optional
import warnings

# ONNX imports
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType

# Scikit-learn imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline


class ONNXModelConverter:
    """Convert scikit-learn models to ONNX format."""
    
    def __init__(self):
        self.supported_models = [
            LinearRegression,
            LogisticRegression,
            RandomForestClassifier,
            RandomForestRegressor
        ]
    
    def convert_sklearn_to_onnx(self, 
                               model: Any, 
                               input_shape: Tuple[int, ...], 
                               model_name: str = "sklearn_model") -> onnx.ModelProto:
        """
        Convert a scikit-learn model to ONNX format.
        
        Args:
            model: Trained scikit-learn model
            input_shape: Shape of input data (features)
            model_name: Name for the ONNX model
            
        Returns:
            ONNX model
        """
        if not any(isinstance(model, supported_type) for supported_type in self.supported_models):
            raise ValueError(f"Model type {type(model)} not supported for ONNX conversion")
        
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, input_shape[1]]))]
        
        # Convert to ONNX
        try:
            onnx_model = convert_sklearn(
                model, 
                initial_types=initial_type,
                target_opset=12  # Use a stable opset version
            )
            
            # Set model name
            onnx_model.graph.name = model_name
            
            return onnx_model
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to ONNX: {str(e)}")
    
    def save_onnx_model(self, onnx_model: onnx.ModelProto, file_path: str) -> None:
        """
        Save ONNX model to file.
        
        Args:
            onnx_model: ONNX model to save
            file_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model
        with open(file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    
    def load_onnx_model(self, file_path: str) -> onnx.ModelProto:
        """
        Load ONNX model from file.
        
        Args:
            file_path: Path to the ONNX model file
            
        Returns:
            Loaded ONNX model
        """
        return onnx.load(file_path)


class ONNXInferenceEngine:
    """ONNX Runtime inference engine."""
    
    def __init__(self, model_path: str):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_names = None
        self.output_names = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the ONNX model for inference."""
        try:
            # Create inference session
            self.session = ort.InferenceSession(self.model_path)
            
            # Get input and output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Input names: {self.input_names}")
            print(f"Output names: {self.output_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Prediction results
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure input data is float32
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Prepare input dictionary
        input_dict = {self.input_names[0]: input_data}
        
        # Run inference
        outputs = self.session.run(self.output_names, input_dict)
        
        return outputs[0]  # Return first output
    
    def predict_batch(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run batch inference on input data.
        
        Args:
            input_data: Batch of input data for prediction
            
        Returns:
            Batch prediction results
        """
        return self.predict(input_data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.session is None:
            return {}
        
        inputs_info = []
        for input_tensor in self.session.get_inputs():
            inputs_info.append({
                'name': input_tensor.name,
                'type': input_tensor.type,
                'shape': input_tensor.shape
            })
        
        outputs_info = []
        for output_tensor in self.session.get_outputs():
            outputs_info.append({
                'name': output_tensor.name,
                'type': output_tensor.type,
                'shape': output_tensor.shape
            })
        
        return {
            'inputs': inputs_info,
            'outputs': outputs_info,
            'providers': self.session.get_providers()
        }


def train_and_convert_model(X_train: pd.DataFrame, 
                           y_train: pd.Series, 
                           model_type: str = "linear_regression",
                           save_path: str = "models/") -> Tuple[Any, str]:
    """
    Train a model and convert it to ONNX format.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model to train
        save_path: Directory to save the ONNX model
        
    Returns:
        Tuple of (trained_model, onnx_model_path)
    """
    # Train the model
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=42)
    elif model_type == "random_forest_regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "random_forest_classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Convert to ONNX
    converter = ONNXModelConverter()
    input_shape = X_train.shape
    onnx_model = converter.convert_sklearn_to_onnx(model, input_shape, model_type)
    
    # Save ONNX model
    onnx_file_path = os.path.join(save_path, f"{model_type}.onnx")
    converter.save_onnx_model(onnx_model, onnx_file_path)
    
    return model, onnx_file_path


def benchmark_inference(model_path: str, 
                       test_data: np.ndarray, 
                       num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark ONNX model inference performance.
    
    Args:
        model_path: Path to ONNX model
        test_data: Test data for benchmarking
        num_runs: Number of inference runs
        
    Returns:
        Performance metrics
    """
    import time
    
    engine = ONNXInferenceEngine(model_path)
    
    # Warm up
    for _ in range(10):
        engine.predict(test_data[:1])
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        engine.predict(test_data[:1])
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times) * 1000  # Convert to milliseconds
    
    return {
        'mean_inference_time_ms': np.mean(times),
        'std_inference_time_ms': np.std(times),
        'min_inference_time_ms': np.min(times),
        'max_inference_time_ms': np.max(times),
        'median_inference_time_ms': np.median(times)
    }
