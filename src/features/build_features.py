"""
Transformer oil temperature feature engineering

This module builds features for transformer oil temperature prediction:
- Basic lag features 
- Rolling statistics
- Load ratios and differences
- Fourier features for seasonality
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_lag_features(df, columns, lag_periods):
    """
    Create lag features for given columns
    
    Args:
        df: DataFrame with datetime index
        columns: List of column names
        lag_periods: List of periods to lag
        
    Returns:
        DataFrame with lag features added
    """
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    df_result = df.sort_index().copy()
    
    # Generate lag features
    for col in columns:
        if col not in df_result.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
            
        for lag in lag_periods:
            df_result[f"{col}_lag_{lag}"] = df_result[col].shift(lag)
    
    return df_result

def create_rolling_features(df, columns, windows, 
                            functions={'mean': np.mean, 'std': np.std}):
    """
    Create rolling window features
    
    Args:
        df: DataFrame with datetime index
        columns: Columns to create features for
        windows: Window sizes (number of time periods)
        functions: Dict of functions to apply to windows
        
    Returns:
        DataFrame with rolling features added
    """
    # Validate inputs
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    df_result = df.sort_index().copy()
    
    # Create features
    for col in columns:
        if col not in df_result.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
            
        for window in windows:
            rolling = df_result[col].rolling(window=window, min_periods=1)
            
            for name, func in functions.items():
                df_result[f"{col}_roll_{window}_{name}"] = rolling.apply(func, raw=True)
    
    return df_result

def create_load_ratios(df):
    """
    Create ratios between useful and useless loads
    
    Args:
        df: DataFrame with load columns
        
    Returns:
        DataFrame with ratio features added
    """
    df_result = df.copy()
    
    # Load pairs (useful/useless)
    pairs = [
        ('HUFL', 'HULL', 'high'),
        ('MUFL', 'MULL', 'mid'),
        ('LUFL', 'LULL', 'low')
    ]
    
    # Create ratios
    for useful, useless, level in pairs:
        if useful not in df.columns or useless not in df.columns:
            continue
            
        ratio_name = f"{level}_load_ratio"
        
        # Avoid division by zero
        mask = df_result[useless] != 0
        df_result[ratio_name] = np.nan
        df_result.loc[mask, ratio_name] = df_result.loc[mask, useful] / df_result.loc[mask, useless]
        
        # Fix infinite values
        df_result[ratio_name] = df_result[ratio_name].replace([np.inf, -np.inf], np.nan)
        max_val = df_result[ratio_name].max() if not pd.isna(df_result[ratio_name].max()) else 100
        df_result[ratio_name] = df_result[ratio_name].fillna(max_val)
    
    # Total ratio
    loads = ['HUFL', 'MUFL', 'LUFL', 'HULL', 'MULL', 'LULL']
    if all(col in df.columns for col in loads):
        useful_sum = df_result['HUFL'] + df_result['MUFL'] + df_result['LUFL']
        useless_sum = df_result['HULL'] + df_result['MULL'] + df_result['LULL']
        
        mask = useless_sum != 0
        df_result['total_load_ratio'] = np.nan
        df_result.loc[mask, 'total_load_ratio'] = useful_sum.loc[mask] / useless_sum.loc[mask]
        
        df_result['total_load_ratio'] = df_result['total_load_ratio'].replace([np.inf, -np.inf], np.nan)
        max_val = df_result['total_load_ratio'].max() if not pd.isna(df_result['total_load_ratio'].max()) else 100
        df_result['total_load_ratio'] = df_result['total_load_ratio'].fillna(max_val)
    
    return df_result

def create_load_diffs(df):
    """
    Create difference features between load levels
    
    Args:
        df: DataFrame with load columns
        
    Returns:
        DataFrame with difference features added
    """
    df_result = df.copy()
    
    # Define load groups
    useful = ['HUFL', 'MUFL', 'LUFL']
    useless = ['HULL', 'MULL', 'LULL']
    
    # Calculate differences between levels
    if all(col in df.columns for col in useful):
        df_result['high_mid_useful_diff'] = df_result['HUFL'] - df_result['MUFL']
        df_result['mid_low_useful_diff'] = df_result['MUFL'] - df_result['LUFL']
        df_result['high_low_useful_diff'] = df_result['HUFL'] - df_result['LUFL']
    
    if all(col in df.columns for col in useless):
        df_result['high_mid_useless_diff'] = df_result['HULL'] - df_result['MULL']
        df_result['mid_low_useless_diff'] = df_result['MULL'] - df_result['LULL']
        df_result['high_low_useless_diff'] = df_result['HULL'] - df_result['LULL']
    
    # Calculate total load
    all_loads = useful + useless
    if all(col in df.columns for col in all_loads):
        df_result['total_load'] = sum(df_result[col] for col in all_loads)
    
    return df_result

# Version 2.0 - Added Fourier features
def create_fourier_features(df):
    """
    Create Fourier features for seasonality
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with Fourier features added
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    df_result = df.copy()
    
    # Calculate hours
    start_year = pd.Timestamp(df.index[0].year, 1, 1)
    hours_of_year = ((df.index - start_year).total_seconds() / 3600).astype(int) % 8760
    hours_of_week = (df.index.dayofweek * 24 + df.index.hour) % 168
    hours_of_day = df.index.hour
    
    # Define periods and orders
    periods = {
        'daily': {'hours': hours_of_day, 'period': 24, 'order': 4},
        'weekly': {'hours': hours_of_week, 'period': 168, 'order': 3},
        'yearly': {'hours': hours_of_year, 'period': 8760, 'order': 5}
    }
    
    # Create features
    for period_name, data in periods.items():
        for k in range(1, data['order'] + 1):
            df_result[f'{period_name}_sin_{k}'] = np.sin(2 * k * np.pi * data['hours'] / data['period'])
            df_result[f'{period_name}_cos_{k}'] = np.cos(2 * k * np.pi * data['hours'] / data['period'])
    
    return df_result

def build_features(df, configs=None):
    """
    Apply all feature engineering techniques
    
    Args:
        df: DataFrame with datetime index
        configs: Optional dictionary with configuration parameters
        
    Returns:
        DataFrame with all features added
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Default configs
    if configs is None:
        configs = {
            'lag': {
                'enabled': True,
                'columns': ['OT', 'HUFL', 'MUFL', 'LUFL'],
                'periods': [1, 2, 3, 6, 12, 24]
            },
            'rolling': {
                'enabled': True,
                'columns': ['OT'],
                'windows': [4, 12, 24, 48]
            },
            'load_ratios': {
                'enabled': True
            },
            'load_diffs': {
                'enabled': True
            },
            'fourier': {
                'enabled': True
            }
        }
    
    # Apply features
    result = df.copy()
    
    if configs['lag']['enabled']:
        columns = [c for c in configs['lag']['columns'] if c in df.columns]
        result = create_lag_features(result, columns, configs['lag']['periods'])
    
    if configs['rolling']['enabled']:
        columns = [c for c in configs['rolling']['columns'] if c in df.columns]
        result = create_rolling_features(result, columns, configs['rolling']['windows'])
    
    if configs['load_ratios']['enabled']:
        result = create_load_ratios(result)
    
    if configs['load_diffs']['enabled']:
        result = create_load_diffs(result)
    
    if configs['fourier']['enabled']:
        result = create_fourier_features(result)
    
    # Handle missing values
    result = result.ffill(limit=12)
    if result.isnull().sum().sum() > 0:
        result = result.interpolate(method='time')
        result = result.bfill()
    
    return result

def process_data(train_path, test_path, out_train_path, out_test_path, **kwargs):
    """
    Process data files with all feature engineering
    
    Args:
        train_path: Path to training data
        test_path: Path to testing data
        out_train_path: Output path for processed training data
        out_test_path: Output path for processed testing data
        **kwargs: Additional arguments for build_features
        
    Returns:
        Tuple of processed train and test DataFrames
    """
    logger.info("Loading data...")
    
    # Load data
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Process data
    logger.info("Creating features...")
    train_processed = build_features(train_df, **kwargs)
    test_processed = build_features(test_df, **kwargs)
    
    # Save results
    os.makedirs(os.path.dirname(out_train_path), exist_ok=True)
    train_processed.to_csv(out_train_path)
    test_processed.to_csv(out_test_path)
    
    logger.info(f"Saved to {out_train_path} and {out_test_path}")
    
    return train_processed, test_processed

if __name__ == "__main__":
    # Example usage
    project_dir = Path(__file__).resolve().parents[2]
    print(f"Project directory: {project_dir}")
    
    # Set paths
    inputs = {
        'train': project_dir / "data" / "preprocessed" / "train_processed.csv",
        'test': project_dir / "data" / "preprocessed" / "test_processed.csv"
    }
    
    outputs = {
        'train': project_dir / "data" / "features" / "train_features.csv",
        'test': project_dir / "data" / "features" / "test_features.csv"
    }
    
    # Create output directory
    os.makedirs(project_dir / "data" / "features", exist_ok=True)
    
    # Run processing
    configs = {
        'lag': {
            'enabled': True,
            'columns': ['OT', 'HUFL', 'MUFL', 'LUFL'],
            'periods': [1, 2, 3, 6, 12, 24]
        },
        'rolling': {
            'enabled': True,
            'columns': ['OT'],
            'windows': [4, 12, 24, 48, 168]
        },
        'load_ratios': {
            'enabled': True
        },
        'load_diffs': {
            'enabled': True
        },
        'fourier': {
            'enabled': True
        }
    }
    
    train_features, test_features = process_data(
        str(inputs['train']),
        str(inputs['test']),
        str(outputs['train']),
        str(outputs['test']),
        configs=configs
    )