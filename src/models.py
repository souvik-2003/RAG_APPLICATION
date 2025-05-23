"""
Machine learning models and related functions.
"""
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Union, List, Tuple, Optional
import os

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a linear regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, 
                        y_train: pd.Series, 
                        is_classification: bool = True, 
                        n_estimators: int = 100,
                        max_depth: Optional[int] = None) -> Union[RandomForestClassifier, RandomForestRegressor]:
    """
    Train a random forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        is_classification: Whether this is a classification task
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        
    Returns:
        Trained random forest model
    """
    if is_classification:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a regression model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a classification model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # For binary classification
    if len(np.unique(y_test)) == 2:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    else:
        # For multiclass
        return {
            'accuracy': accuracy
        }


def save_model(model: Any, file_path: str) -> None:
    """
    Save a trained model to a file.
    
    Args:
        model: Trained model
        file_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path: str) -> Any:
    """
    Load a model from a file.
    
    Args:
        file_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    return model
