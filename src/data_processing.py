"""
Data processing utilities for the project.
"""
import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file (CSV, Excel, etc.) into a pandas DataFrame.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame containing the data
    """
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        return pd.read_csv(file_path)
    elif ext.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values, outliers, etc.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Handle missing values
    df_cleaned = df.copy()
    
    # Drop rows with all NaN values
    df_cleaned = df_cleaned.dropna(how='all')
    
    # Fill numeric columns with their median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # Fill categorical columns with their mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if not df_cleaned[col].mode().empty:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")
    
    return df_cleaned


def split_data(df: pd.DataFrame, 
               target_column: str, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of the data to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def preprocess_features(df: pd.DataFrame, 
                         categorical_cols: Optional[List[str]] = None, 
                         numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Preprocess features for machine learning.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        
    Returns:
        Preprocessed DataFrame
    """
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # If not specified, automatically detect column types
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Apply preprocessing
    preprocessed_array = preprocessor.fit_transform(df)
    
    # Convert to DataFrame with feature names (simplified for demonstration)
    preprocessed_df = pd.DataFrame(
        preprocessed_array,
        index=df.index
    )
    
    return preprocessed_df
