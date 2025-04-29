import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lag_periods: List[int]
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Creating lag features for {len(columns)} columns with {len(lag_periods)} lag periods")
    
    df_with_lags = df.sort_index().copy()
    
    for col in columns:
        if col not in df_with_lags.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        for lag in lag_periods:
            lag_name = f"{col}_lag_{lag}"
            df_with_lags[lag_name] = df_with_lags[col].shift(lag)
            logger.debug(f"Created lag feature: {lag_name}")
    
    logger.info(f"Created {len(columns) * len(lag_periods)} lag features")
    return df_with_lags

def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    functions: Dict[str, callable] = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max}
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Creating rolling features for {len(columns)} columns with {len(windows)} windows")
    
    df_with_rolling = df.sort_index().copy()
    
    for col in columns:
        if col not in df_with_rolling.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        for window in windows:
            rolling = df_with_rolling[col].rolling(window=window, min_periods=1)
            
            for func_name, func in functions.items():
                feature_name = f"{col}_rolling_{window}_{func_name}"
                df_with_rolling[feature_name] = rolling.apply(func, raw=True)
                logger.debug(f"Created rolling feature: {feature_name}")
    
    total_features = len(columns) * len(windows) * len(functions)
    logger.info(f"Created {total_features} rolling features")
    return df_with_rolling

def create_load_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating load ratio features")
    
    df_with_ratios = df.copy()
    
    load_pairs = [
        ('HUFL', 'HULL', 'high'),
        ('MUFL', 'MULL', 'middle'),
        ('LUFL', 'LULL', 'low')
    ]
    
    for useful, useless, level in load_pairs:
        if useful not in df.columns or useless not in df.columns:
            logger.warning(f"Columns '{useful}' and/or '{useless}' not found in DataFrame, skipping")
            continue
            
        ratio_name = f"{level}_load_ratio"
        
        mask = df_with_ratios[useless] != 0
        df_with_ratios[ratio_name] = np.nan
        df_with_ratios.loc[mask, ratio_name] = df_with_ratios.loc[mask, useful] / df_with_ratios.loc[mask, useless]
        
        temp = df_with_ratios[ratio_name].replace([np.inf, -np.inf], np.nan)
        max_val = temp.max() if temp.max() > 0 else 100
        df_with_ratios[ratio_name] = temp.fillna(max_val)
        
        logger.debug(f"Created ratio feature: {ratio_name}")
    
    if all(col in df.columns for col in ['HUFL', 'MUFL', 'LUFL', 'HULL', 'MULL', 'LULL']):
        total_useful = df_with_ratios['HUFL'] + df_with_ratios['MUFL'] + df_with_ratios['LUFL']
        total_useless = df_with_ratios['HULL'] + df_with_ratios['MULL'] + df_with_ratios['LULL']
        
        mask = total_useless != 0
        df_with_ratios['total_load_ratio'] = np.nan
        df_with_ratios.loc[mask, 'total_load_ratio'] = total_useful.loc[mask] / total_useless.loc[mask]
        
        temp = df_with_ratios['total_load_ratio'].replace([np.inf, -np.inf], np.nan)
        max_val = temp.max() if temp.max() > 0 else 100
        df_with_ratios['total_load_ratio'] = temp.fillna(max_val)
        
        logger.debug("Created total load ratio feature")
    
    logger.info("Created load ratio features")
    return df_with_ratios

def create_load_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating load difference features")
    
    df_with_diffs = df.copy()
    
    useful_loads = ['HUFL', 'MUFL', 'LUFL']
    useless_loads = ['HULL', 'MULL', 'LULL']
    
    if all(col in df.columns for col in useful_loads):
        df_with_diffs['high_mid_useful_diff'] = df_with_diffs['HUFL'] - df_with_diffs['MUFL']
        df_with_diffs['mid_low_useful_diff'] = df_with_diffs['MUFL'] - df_with_diffs['LUFL']
        df_with_diffs['high_low_useful_diff'] = df_with_diffs['HUFL'] - df_with_diffs['LUFL']
        logger.debug("Created useful load difference features")
    
    if all(col in df.columns for col in useless_loads):
        df_with_diffs['high_mid_useless_diff'] = df_with_diffs['HULL'] - df_with_diffs['MULL']
        df_with_diffs['mid_low_useless_diff'] = df_with_diffs['MULL'] - df_with_diffs['LULL']
        df_with_diffs['high_low_useless_diff'] = df_with_diffs['HULL'] - df_with_diffs['LULL']
        logger.debug("Created useless load difference features")
    
    all_loads = useful_loads + useless_loads
    if all(col in df.columns for col in all_loads):
        df_with_diffs['total_load'] = sum([df_with_diffs[col] for col in all_loads])
        logger.debug("Created total load feature")
    
    logger.info("Created load difference features")
    return df_with_diffs

def create_fourier_features(
    df: pd.DataFrame,
    periods: Dict[str, int] = {'daily': 24, 'weekly': 168, 'yearly': 8760},
    fourier_order: Dict[str, int] = {'daily': 4, 'weekly': 3, 'yearly': 5}
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info("Creating Fourier features")
    
    df_with_fourier = df.copy()
    
    start_of_year = pd.Timestamp(df.index[0].year, 1, 1)
    hours_of_year = ((df.index - start_of_year).total_seconds() / 3600).astype(int) % 8760
    
    hours_of_week = (df.index.dayofweek * 24 + df.index.hour) % 168
    
    hours_of_day = df.index.hour
    
    for period_name, period in periods.items():
        order = fourier_order[period_name]
        
        if period_name == 'daily':
            hour_var = hours_of_day
        elif period_name == 'weekly':
            hour_var = hours_of_week
        elif period_name == 'yearly':
            hour_var = hours_of_year
        else:
            logger.warning(f"Unknown period: {period_name}, skipping")
            continue
        
        for k in range(1, order + 1):
            df_with_fourier[f'{period_name}_sin_{k}'] = np.sin(2 * k * np.pi * hour_var / period)
            df_with_fourier[f'{period_name}_cos_{k}'] = np.cos(2 * k * np.pi * hour_var / period)
            logger.debug(f"Created Fourier feature: {period_name}_sin_{k} and {period_name}_cos_{k}")
    
    total_features = sum([2 * fourier_order[period] for period in periods])
    logger.info(f"Created {total_features} Fourier features")
    return df_with_fourier

def build_feature_set(
    df: pd.DataFrame,
    include_lag: bool = True,
    include_rolling: bool = True,
    include_load_ratio: bool = True,
    include_load_diff: bool = True,
    include_fourier: bool = True,
    lag_columns: Optional[List[str]] = None,
    lag_periods: Optional[List[int]] = None,
    rolling_columns: Optional[List[str]] = None,
    rolling_windows: Optional[List[int]] = None,
    fourier_periods: Optional[Dict[str, int]] = None,
    fourier_order: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    logger.info("Building comprehensive feature set")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    result_df = df.copy()
    
    if lag_columns is None:
        lag_columns = ['OT', 'HUFL', 'MUFL', 'LUFL']
        lag_columns = [col for col in lag_columns if col in result_df.columns]
    
    if lag_periods is None:
        lag_periods = [1, 2, 3, 4, 5, 6, 12, 24]
    
    if rolling_columns is None:
        rolling_columns = ['OT']
        rolling_columns = [col for col in rolling_columns if col in result_df.columns]
    
    if rolling_windows is None:
        rolling_windows = [4, 12, 24, 48, 168]
    
    if fourier_periods is None:
        fourier_periods = {'daily': 24, 'weekly': 168, 'yearly': 8760}
    
    if fourier_order is None:
        fourier_order = {'daily': 4, 'weekly': 3, 'yearly': 5}
    
    if include_lag and lag_columns:
        result_df = create_lag_features(result_df, lag_columns, lag_periods)
    
    if include_rolling and rolling_columns:
        result_df = create_rolling_features(result_df, rolling_columns, rolling_windows)
    
    if include_load_ratio:
        result_df = create_load_ratio_features(result_df)
    
    if include_load_diff:
        result_df = create_load_difference_features(result_df)
    
    if include_fourier:
        result_df = create_fourier_features(result_df, fourier_periods, fourier_order)
    
    result_df = result_df.ffill(limit=12)
    
    if result_df.isnull().sum().sum() > 0:
        logger.info("Using time interpolation for remaining missing values")
        result_df = result_df.interpolate(method='time')
        
        if result_df.isnull().sum().sum() > 0:
            result_df = result_df.bfill()
            logger.info("Used backward fill for missing values at beginning of time series")
    
    logger.info(f"Feature set built, shape: {result_df.shape}")
    return result_df

def engineer_features(
    input_train_path: str,
    input_test_path: str,
    output_train_path: str,
    output_test_path: str,
    **feature_args
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Starting feature engineering process")
    
    logger.info(f"Loading preprocessed training data from {input_train_path}")
    try:
        train_df = pd.read_csv(input_train_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded training data with shape {train_df.shape}")
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise
    
    logger.info(f"Loading preprocessed testing data from {input_test_path}")
    try:
        test_df = pd.read_csv(input_test_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded testing data with shape {test_df.shape}")
    except Exception as e:
        logger.error(f"Error loading testing data: {str(e)}")
        raise
    
    logger.info("Engineering features for training data")
    train_features = build_feature_set(train_df, **feature_args)
    
    logger.info("Engineering features for testing data")
    test_features = build_feature_set(test_df, **feature_args)
    
    logger.info(f"Saving feature engineered training data to {output_train_path}")
    output_dir = os.path.dirname(output_train_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    try:
        train_features.to_csv(output_train_path)
        logger.info(f"Successfully saved training data with shape {train_features.shape}")
    except Exception as e:
        logger.error(f"Error saving training data: {str(e)}")
        raise
    
    logger.info(f"Saving feature engineered testing data to {output_test_path}")
    try:
        test_features.to_csv(output_test_path)
        logger.info(f"Successfully saved testing data with shape {test_features.shape}")
    except Exception as e:
        logger.error(f"Error saving testing data: {str(e)}")
        raise
    
    logger.info("Feature engineering process completed")
    return train_features, test_features

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    
    print(f"Project directory: {project_dir}")
    
    input_train_path = project_dir / "data" / "preprocessed" / "train_processed.csv"
    input_test_path = project_dir / "data" / "preprocessed" / "test_processed.csv"
    output_train_path = project_dir / "data" / "features" / "train_features.csv"
    output_test_path = project_dir / "data" / "features" / "test_features.csv"
    
    output_dir = project_dir / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input train path: {input_train_path}")
    print(f"Input test path: {input_test_path}")
    print(f"Output train path: {output_train_path}")
    print(f"Output test path: {output_test_path}")
    
    train_features, test_features = engineer_features(
        input_train_path=str(input_train_path),
        input_test_path=str(input_test_path),
        output_train_path=str(output_train_path),
        output_test_path=str(output_test_path),
        include_lag=True,
        lag_columns=['OT', 'HUFL', 'MUFL', 'LUFL'],
        lag_periods=[1, 2, 3, 4, 6, 12, 24],
        include_rolling=True,
        rolling_columns=['OT'],
        rolling_windows=[4, 12, 24, 48, 168],
        include_load_ratio=True,
        include_load_diff=True,
        include_fourier=True
    )