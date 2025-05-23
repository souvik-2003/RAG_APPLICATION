"""
Visualization utilities for the project.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union


def plot_distribution(data: pd.Series, title: str = '', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of a variable.
    
    Args:
        data: Series to plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if data.dtype.kind in 'ifc':
        # Numeric data
        sns.histplot(data, kde=True, ax=ax)
    else:
        # Categorical data
        sns.countplot(y=data, ax=ax)
    
    ax.set_title(title if title else f'Distribution of {data.name}')
    ax.set_ylabel('Frequency')
    
    if data.dtype.kind in 'ifc':
        ax.set_xlabel(data.name)
    else:
        ax.set_xlabel('Count')
        ax.set_ylabel(data.name)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot correlation matrix of numeric variables.
    
    Args:
        data: DataFrame containing the variables
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr = numeric_data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    
    return fig


def plot_feature_importance(model: Any, feature_names: List[str], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names: List of feature names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot feature importances
    sns.barplot(x=importances[indices][:20], y=[feature_names[i] for i in indices][:20], ax=ax)
    
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    
    return fig


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Plot predicted vs actual values for regression models.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted Values')
    
    plt.tight_layout()
    return fig
