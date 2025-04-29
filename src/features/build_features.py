"""
Functions for building features from preprocessed transformer oil temperature data.

This module contains functions for creating various types of features:
1. Lag features - previous values of target and predictors
2. Rolling window features - statistics over time windows
3. Load ratio features - ratios between useful and useless loads
4. Load difference features - differences between load levels
5. Fourier features - for better capturing seasonality

Based on the EDA findings, these features should help capture:
- Strong autocorrelation in oil temperature
- Daily and seasonal patterns
- Relationships between different load variables
- Temporal dependencies in the data
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_temperature_rate_features(
    df: pd.DataFrame,
    temperature_column: str = 'OT',
    windows: List[int] = [1, 2, 3, 6, 12, 24, 48]
) -> pd.DataFrame:
    """
    Create features that capture the rate of change in temperature.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    temperature_column : str, optional
        Name of temperature column, by default 'OT'
    windows : List[int], optional
        Windows for calculating rate of change, by default [1, 2, 3, 6, 12, 24, 48]
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional rate of change features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Creating temperature rate of change features for {temperature_column}")
    
    df_with_rates = df.copy()
    
    if temperature_column not in df.columns:
        logger.warning(f"Temperature column '{temperature_column}' not found in DataFrame")
        return df_with_rates
    
    # Create rate of change features for different time windows
    for window in windows:
        # Simple difference (change per time step)
        df_with_rates[f'{temperature_column}_diff_{window}'] = df[temperature_column].diff(window)
        
        # Rate of change (change per unit time)
        df_with_rates[f'{temperature_column}_roc_{window}'] = df[temperature_column].diff(window) / window
        
        # Percentage change
        df_with_rates[f'{temperature_column}_pct_{window}'] = df[temperature_column].pct_change(window) * 100
        
        logger.debug(f"Created rate features with window {window}")
    
    # Create acceleration features (rate of change of the rate of change)
    for window in windows[:-1]:  # Use smaller windows for acceleration
        df_with_rates[f'{temperature_column}_accel_{window}'] = df_with_rates[f'{temperature_column}_roc_{window}'].diff(window)
    
    logger.info(f"Created {len(windows) * 3 + len(windows) - 1} temperature rate features")
    return df_with_rates

def create_extreme_temperature_features(
    df: pd.DataFrame,
    temperature_column: str = 'OT',
    threshold_quantile: float = 0.9,
    windows: List[int] = [24, 48, 72, 168]  # 1, 2, 3, 7 days
) -> pd.DataFrame:
    """
    Create features specifically designed to help predict extreme temperatures.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    temperature_column : str, optional
        Name of temperature column, by default 'OT'
    threshold_quantile : float, optional
        Quantile threshold for extreme temperature, by default 0.9
    windows : List[int], optional
        Windows for calculating features, by default [24, 48, 72, 168]
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional extreme temperature features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info("Creating extreme temperature features")
    
    # Instead of modifying DataFrame directly, collect all new features in a dictionary
    new_features = {}
    
    if temperature_column not in df.columns:
        logger.warning(f"Temperature column '{temperature_column}' not found in DataFrame")
        return df.copy()
    
    # Calculate extreme temperature threshold
    threshold = df[temperature_column].quantile(threshold_quantile)
    logger.info(f"Extreme temperature threshold ({threshold_quantile} quantile): {threshold:.2f}")
    
    # Create binary indicator for extreme temperatures
    new_features[f'{temperature_column}_extreme'] = (df[temperature_column] >= threshold).astype(int)
    
    # Track time since last extreme temperature
    extreme_events = df.index[df[temperature_column] >= threshold]
    
    if len(extreme_events) > 0:
        # Initialize with large value
        hours_since_extreme = np.full(len(df), len(df))  # Initialize with large value
        
        # For each time point, calculate hours since most recent extreme event
        for i, current_time in enumerate(df.index):
            # Find the most recent extreme event before current time
            past_extremes = extreme_events[extreme_events < current_time]
            
            if len(past_extremes) > 0:
                most_recent = past_extremes[-1]
                hours_diff = (current_time - most_recent).total_seconds() / 3600
                hours_since_extreme[i] = hours_diff
        
        new_features['hours_since_extreme'] = hours_since_extreme
    
    # Calculate moving max and range features
    for window in windows:
        # Maximum temperature in past window
        new_features[f'{temperature_column}_max_{window}'] = df[temperature_column].rolling(window=window, min_periods=1).max()
        
        # Temperature range in past window
        new_features[f'{temperature_column}_range_{window}'] = (
            df[temperature_column].rolling(window=window, min_periods=1).max() - 
            df[temperature_column].rolling(window=window, min_periods=1).min()
        )
        
        # Count of extreme events in past window
        extreme_indicator = (df[temperature_column] >= threshold).astype(int)
        new_features[f'extreme_count_{window}'] = extreme_indicator.rolling(window=window, min_periods=1).sum()
    
    # Exponentially weighted features
    for span in [12, 24, 48]:
        # EWM on temperature
        ewm_mean = df[temperature_column].ewm(span=span).mean()
        new_features[f'{temperature_column}_ewm_{span}'] = ewm_mean
        
        # EWM on temperature difference from EWM
        new_features[f'{temperature_column}_ewm_diff_{span}'] = df[temperature_column] - ewm_mean
    
    # Create a DataFrame with all new features
    new_features_df = pd.DataFrame(new_features, index=df.index)
    
    # Concatenate with original DataFrame
    df_with_extremes = pd.concat([df.copy(), new_features_df], axis=1)
    
    logger.info(f"Created extreme temperature features")
    return df_with_extremes

def create_trend_reversal_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [6, 12, 24, 48]
) -> pd.DataFrame:
    """
    Create features that help predict trend reversals in time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    columns : List[str]
        Columns to create trend reversal features for
    windows : List[int], optional
        Windows for calculating trend features, by default [6, 12, 24, 48]
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional trend reversal features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Creating trend reversal features for {len(columns)} columns")
    
    # Instead of modifying the dataframe directly, collect all new features
    # and concatenate them at the end
    new_features = {}
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
        
        for window in windows:
            # Calculate rolling mean
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            
            # Trend direction (1 for up, -1 for down, 0 for flat)
            trend_dir_name = f'{col}_trend_dir_{window}'
            new_features[trend_dir_name] = np.sign(rolling_mean.diff())
            
            # Trend strength (difference from longer-term mean)
            long_window = window * 2
            long_mean = df[col].rolling(window=long_window, min_periods=1).mean()
            new_features[f'{col}_trend_str_{window}'] = rolling_mean - long_mean
            
            # Trend duration (how long current trend has persisted)
            trend_changes = np.sign(rolling_mean.diff()).diff().fillna(0) != 0
            trend_change_points = np.where(trend_changes)[0]
            
            # Initialize trend duration array
            trend_duration = np.zeros(len(df))
            
            # Calculate duration for each point
            for i in range(len(df)):
                # Find most recent change point before current position
                prev_changes = trend_change_points[trend_change_points < i]
                
                if len(prev_changes) > 0:
                    most_recent = prev_changes[-1]
                    trend_duration[i] = i - most_recent
                else:
                    trend_duration[i] = i  # No previous change point
            
            new_features[f'{col}_trend_dur_{window}'] = trend_duration
            
            # Calculate direction changes separately using the new feature we just created
            # Instead of trying to access it from the original DataFrame
            trend_dir_values = new_features[trend_dir_name]
            
            # For the rolling calculation on our new column, use pandas Series directly
            direction_changes = pd.Series(trend_dir_values).rolling(window=window).apply(
                lambda x: ((x[:-1] * x[1:]) < 0).sum() if len(x) > 1 else 0, 
                raw=True
            )
            new_features[f'{col}_oscillation_{window}'] = direction_changes.values
            
            logger.debug(f"Created trend features for '{col}' with window {window}")
    
    # Create a DataFrame with all new features
    new_features_df = pd.DataFrame(new_features, index=df.index)
    
    # Concatenate with the original DataFrame
    df_with_trends = pd.concat([df.copy(), new_features_df], axis=1)
    
    logger.info(f"Created trend reversal features")
    return df_with_trends

def create_enhanced_load_features(
    df: pd.DataFrame,
    useful_load_columns: List[str] = ['HUFL', 'MUFL', 'LUFL'],
    useless_load_columns: List[str] = ['HULL', 'MULL', 'LULL'],
    windows: List[int] = [4, 12, 24, 48]
) -> pd.DataFrame:
    """
    Create enhanced features from load columns with focus on relationship to temperature.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    useful_load_columns : List[str], optional
        Columns representing useful loads, by default ['HUFL', 'MUFL', 'LUFL']
    useless_load_columns : List[str], optional
        Columns representing useless loads, by default ['HULL', 'MULL', 'LULL']
    windows : List[int], optional
        Windows for calculating features, by default [4, 12, 24, 48]
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional enhanced load features
    """
    logger.info("Creating enhanced load features")
    
    df_with_loads = df.copy()
    
    # Check if required columns exist
    missing_useful = [col for col in useful_load_columns if col not in df.columns]
    missing_useless = [col for col in useless_load_columns if col not in df.columns]
    
    if missing_useful or missing_useless:
        missing = missing_useful + missing_useless
        logger.warning(f"Some load columns are missing: {missing}")
        
        # Filter to only include available columns
        useful_load_columns = [col for col in useful_load_columns if col in df.columns]
        useless_load_columns = [col for col in useless_load_columns if col in df.columns]
    
    # If we have both useful and useless loads
    if useful_load_columns and useless_load_columns:
        # Create total loads
        if len(useful_load_columns) > 0:
            df_with_loads['total_useful_load'] = df[useful_load_columns].sum(axis=1)
        
        if len(useless_load_columns) > 0:
            df_with_loads['total_useless_load'] = df[useless_load_columns].sum(axis=1)
        
        # Total load and ratio if both are available
        if 'total_useful_load' in df_with_loads.columns and 'total_useless_load' in df_with_loads.columns:
            df_with_loads['total_load'] = df_with_loads['total_useful_load'] + df_with_loads['total_useless_load']
            
            # Avoid division by zero
            mask = df_with_loads['total_useless_load'] != 0
            df_with_loads['load_efficiency'] = np.nan
            df_with_loads.loc[mask, 'load_efficiency'] = (
                df_with_loads.loc[mask, 'total_useful_load'] / df_with_loads.loc[mask, 'total_useless_load']
            )
            
            # Replace infinities and NaNs with a high value
            df_with_loads['load_efficiency'] = df_with_loads['load_efficiency'].replace([np.inf, -np.inf], np.nan)
            df_with_loads['load_efficiency'] = df_with_loads['load_efficiency'].fillna(
                df_with_loads['load_efficiency'].max() if df_with_loads['load_efficiency'].max() > 0 else 100
            )
    
    # Create load volatility features
    all_load_columns = useful_load_columns + useless_load_columns
    
    for col in all_load_columns:
        for window in windows:
            # Load volatility (standard deviation over window)
            df_with_loads[f'{col}_volatility_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            
            # Load momentum (rate of change)
            df_with_loads[f'{col}_momentum_{window}'] = df[col].diff(window) / window
    
    # Create cross-correlations between loads if we have both OT and load columns
    if 'OT' in df.columns:
        for col in all_load_columns:
            for window in windows:
                # Calculate rolling correlation with temperature
                df_with_loads[f'{col}_temp_corr_{window}'] = (
                    df[['OT', col]].rolling(window=window, min_periods=window//2)
                    .corr().unstack()[col]['OT']  # Extract correlation between col and OT
                )
    
    # Create load pattern features (variations in load throughout day)
    if isinstance(df.index, pd.DatetimeIndex):
        # Group by hour of day
        hour_groups = df.groupby(df.index.hour)
        
        # Calculate average load for each hour
        for col in all_load_columns:
            hourly_means = hour_groups[col].mean()
            
            # Create features showing deviation from typical load for this hour
            for hour in range(24):
                if hour in hourly_means.index:
                    typical_load = hourly_means[hour]
                    
                    # Create mask for this hour
                    hour_mask = df.index.hour == hour
                    
                    # Add deviation from typical load for this hour
                    col_name = f'{col}_dev_from_hour_{hour}'
                    df_with_loads[col_name] = np.nan
                    df_with_loads.loc[hour_mask, col_name] = df.loc[hour_mask, col] - typical_load
    
    logger.info(f"Created enhanced load features")
    return df_with_loads

def create_enhanced_fourier_features(
    df: pd.DataFrame,
    periods: Dict[str, int] = {'daily': 24, 'weekly': 168, 'yearly': 8760},
    fourier_orders: Dict[str, int] = {'daily': 6, 'weekly': 4, 'yearly': 8}
) -> pd.DataFrame:
    """
    Create enhanced Fourier features with higher orders for better seasonality capture.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    periods : Dict[str, int], optional
        Dictionary mapping period names to number of hours, by default
        {'daily': 24, 'weekly': 168, 'yearly': 8760}
    fourier_orders : Dict[str, int], optional
        Dictionary mapping period names to Fourier order, by default
        {'daily': 6, 'weekly': 4, 'yearly': 8}
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional enhanced Fourier features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info("Creating enhanced Fourier features with higher orders")
    
    df_with_fourier = df.copy()
    
    # Create hour of year feature (0 to 8759)
    start_of_year = pd.Timestamp(df.index[0].year, 1, 1)
    hours_of_year = ((df.index - start_of_year).total_seconds() / 3600).astype(int) % 8760
    
    # Create hour of week feature (0 to 167)
    hours_of_week = (df.index.dayofweek * 24 + df.index.hour) % 168
    
    # Create hour of day feature (0 to 23)
    hours_of_day = df.index.hour
    
    # Create Fourier features for each period with higher orders
    for period_name, period in periods.items():
        order = fourier_orders[period_name]
        
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
    
    # Add interaction features between Fourier components
    # These help capture more complex seasonal patterns
    for period1 in periods.keys():
        for period2 in periods.keys():
            if period1 != period2:
                # Create interaction between first harmonic of each period
                df_with_fourier[f'{period1}_sin_1_{period2}_sin_1'] = (
                    df_with_fourier[f'{period1}_sin_1'] * df_with_fourier[f'{period2}_sin_1']
                )
                df_with_fourier[f'{period1}_cos_1_{period2}_cos_1'] = (
                    df_with_fourier[f'{period1}_cos_1'] * df_with_fourier[f'{period2}_cos_1']
                )
    
    total_features = sum([2 * fourier_orders[period] for period in periods]) + 4  # Basic + interactions
    logger.info(f"Created {total_features} enhanced Fourier features")
    return df_with_fourier

def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lag_periods: List[int]
) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    columns : List[str]
        List of columns to create lag features for
    lag_periods : List[int]
        List of lag periods to create
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag features
        
    Notes
    -----
    - Based on EDA findings, oil temperature shows strong autocorrelation
    - Lag features help capture temporal dependencies in the data
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Creating lag features for {len(columns)} columns with {len(lag_periods)} lag periods")
    
    # Sort by datetime index to ensure correct lag calculation
    df_with_lags = df.sort_index().copy()
    
    # Create lag features
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
    """
    Create rolling window features for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    columns : List[str]
        List of columns to create rolling features for
    windows : List[int]
        List of window sizes (in time steps)
    functions : Dict[str, callable], optional
        Dictionary mapping function names to functions, by default 
        {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max}
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling features
        
    Notes
    -----
    - Rolling statistics capture recent trends and variability
    - Particularly useful for the oil temperature variable
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Creating rolling features for {len(columns)} columns with {len(windows)} windows")
    
    # Sort by datetime index to ensure correct window calculation
    df_with_rolling = df.sort_index().copy()
    
    # Create rolling features
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
    """
    Create ratio features between useful and useless loads.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with load columns: HUFL, HULL, MUFL, MULL, LUFL, LULL
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional load ratio features
        
    Notes
    -----
    - EDA showed that the ratio between useful and useless loads is significant
    - These ratios can help understand the efficiency of the transformer
    """
    logger.info("Creating load ratio features")
    
    df_with_ratios = df.copy()
    
    # Define pairs of useful and useless loads
    load_pairs = [
        ('HUFL', 'HULL', 'high'),
        ('MUFL', 'MULL', 'middle'),
        ('LUFL', 'LULL', 'low')
    ]
    
    # Create ratio features
    for useful, useless, level in load_pairs:
        if useful not in df.columns or useless not in df.columns:
            logger.warning(f"Columns '{useful}' and/or '{useless}' not found in DataFrame, skipping")
            continue
            
        ratio_name = f"{level}_load_ratio"
        
        # Avoid division by zero
        mask = df_with_ratios[useless] != 0
        df_with_ratios[ratio_name] = np.nan
        df_with_ratios.loc[mask, ratio_name] = df_with_ratios.loc[mask, useful] / df_with_ratios.loc[mask, useless]
        
        # Replace infinities with NaN and fill with a large value - FIXED INPLACE OPERATIONS
        # Instead of using inplace=True, assign the result back to the column
        temp = df_with_ratios[ratio_name].replace([np.inf, -np.inf], np.nan)
        max_val = temp.max() if temp.max() > 0 else 100
        df_with_ratios[ratio_name] = temp.fillna(max_val)
        
        logger.debug(f"Created ratio feature: {ratio_name}")
    
    # Create total useful to useless load ratio
    if all(col in df.columns for col in ['HUFL', 'MUFL', 'LUFL', 'HULL', 'MULL', 'LULL']):
        total_useful = df_with_ratios['HUFL'] + df_with_ratios['MUFL'] + df_with_ratios['LUFL']
        total_useless = df_with_ratios['HULL'] + df_with_ratios['MULL'] + df_with_ratios['LULL']
        
        mask = total_useless != 0
        df_with_ratios['total_load_ratio'] = np.nan
        df_with_ratios.loc[mask, 'total_load_ratio'] = total_useful.loc[mask] / total_useless.loc[mask]
        
        # Replace infinities with NaN and fill with a large value - FIXED INPLACE OPERATIONS
        temp = df_with_ratios['total_load_ratio'].replace([np.inf, -np.inf], np.nan)
        max_val = temp.max() if temp.max() > 0 else 100
        df_with_ratios['total_load_ratio'] = temp.fillna(max_val)
        
        logger.debug("Created total load ratio feature")
    
    logger.info("Created load ratio features")
    return df_with_ratios

def create_load_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create difference features between different load levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with load columns: HUFL, HULL, MUFL, MULL, LUFL, LULL
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional load difference features
        
    Notes
    -----
    - EDA indicated that changes in load values correlate with changes in temperature
    - Differences between load levels can help capture load distribution
    """
    logger.info("Creating load difference features")
    
    df_with_diffs = df.copy()
    
    # Define load levels
    useful_loads = ['HUFL', 'MUFL', 'LUFL']
    useless_loads = ['HULL', 'MULL', 'LULL']
    
    # Create differences between adjacent load levels for useful loads
    if all(col in df.columns for col in useful_loads):
        df_with_diffs['high_mid_useful_diff'] = df_with_diffs['HUFL'] - df_with_diffs['MUFL']
        df_with_diffs['mid_low_useful_diff'] = df_with_diffs['MUFL'] - df_with_diffs['LUFL']
        df_with_diffs['high_low_useful_diff'] = df_with_diffs['HUFL'] - df_with_diffs['LUFL']
        logger.debug("Created useful load difference features")
    
    # Create differences between adjacent load levels for useless loads
    if all(col in df.columns for col in useless_loads):
        df_with_diffs['high_mid_useless_diff'] = df_with_diffs['HULL'] - df_with_diffs['MULL']
        df_with_diffs['mid_low_useless_diff'] = df_with_diffs['MULL'] - df_with_diffs['LULL']
        df_with_diffs['high_low_useless_diff'] = df_with_diffs['HULL'] - df_with_diffs['LULL']
        logger.debug("Created useless load difference features")
    
    # Calculate sum of all loads (total load on transformer)
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
    """
    Create Fourier features to better capture periodicity in time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    periods : Dict[str, int], optional
        Dictionary mapping period names to number of hours, by default
        {'daily': 24, 'weekly': 168, 'yearly': 8760}
    fourier_order : Dict[str, int], optional
        Dictionary mapping period names to Fourier order, by default
        {'daily': 4, 'weekly': 3, 'yearly': 5}
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional Fourier features
        
    Notes
    -----
    - Fourier features provide a more flexible way to model seasonality
    - Higher orders capture more complex seasonal patterns
    - Particularly useful for daily, weekly, and yearly patterns in temperature
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info("Creating Fourier features")
    
    df_with_fourier = df.copy()
    
    # Create hour of year feature (0 to 8759)
    start_of_year = pd.Timestamp(df.index[0].year, 1, 1)
    hours_of_year = ((df.index - start_of_year).total_seconds() / 3600).astype(int) % 8760
    
    # Create hour of week feature (0 to 167)
    hours_of_week = (df.index.dayofweek * 24 + df.index.hour) % 168
    
    # Create hour of day feature (0 to 23)
    hours_of_day = df.index.hour
    
    # Create Fourier features for each period
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
    include_temperature_rates: bool = True,
    include_extreme_temp: bool = True,
    include_trend_reversal: bool = True,
    include_enhanced_load: bool = True,
    lag_columns: Optional[List[str]] = None,
    lag_periods: Optional[List[int]] = None,
    rolling_columns: Optional[List[str]] = None,
    rolling_windows: Optional[List[int]] = None,
    fourier_periods: Optional[Dict[str, int]] = None,
    fourier_order: Optional[Dict[str, int]] = None,
    max_gap_limit: int = 24
) -> pd.DataFrame:
    """
    Build a comprehensive feature set by applying multiple feature engineering techniques.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    include_lag : bool, optional
        Whether to include lag features, by default True
    include_rolling : bool, optional
        Whether to include rolling features, by default True
    include_load_ratio : bool, optional
        Whether to include load ratio features, by default True
    include_load_diff : bool, optional
        Whether to include load difference features, by default True
    include_fourier : bool, optional
        Whether to include Fourier features, by default True
    include_temperature_rates : bool, optional
        Whether to include temperature rate features, by default True
    include_extreme_temp : bool, optional
        Whether to include extreme temperature features, by default True
    include_trend_reversal : bool, optional
        Whether to include trend reversal features, by default True
    include_enhanced_load : bool, optional
        Whether to include enhanced load features, by default True
    lag_columns : List[str], optional
        List of columns to create lag features for, by default None
        (uses ['OT', 'HUFL', 'MUFL', 'LUFL'] if None)
    lag_periods : List[int], optional
        List of lag periods to create, by default None
        (uses [1, 2, 3, 4, 5, 6, 12, 24] if None)
    rolling_columns : List[str], optional
        List of columns to create rolling features for, by default None
        (uses ['OT'] if None)
    rolling_windows : List[int], optional
        List of window sizes (in time steps), by default None
        (uses [4, 12, 24, 48, 168] if None)
    fourier_periods : Dict[str, int], optional
        Dictionary mapping period names to number of hours, by default None
        (uses {'daily': 24, 'weekly': 168, 'yearly': 8760} if None)
    fourier_order : Dict[str, int], optional
        Dictionary mapping period names to Fourier order, by default None
        (uses {'daily': 4, 'weekly': 3, 'yearly': 5} if None)
    max_gap_limit : int, optional
        Maximum size of gaps (in time steps) to fill using interpolation, by default 24
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all selected features
    """
    logger.info("Building comprehensive feature set with enhanced features")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Check for NaN values before starting feature engineering
    initial_nan_count = df.isnull().sum().sum()
    if initial_nan_count > 0:
        logger.warning(f"Input data contains {initial_nan_count} NaN values before feature engineering")
        missing_by_col = df.isnull().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0]
        logger.warning(f"Columns with missing values: \n{cols_with_missing}")
    
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Set default values
    if lag_columns is None:
        lag_columns = ['OT', 'HUFL', 'MUFL', 'LUFL']
        # Filter to only include columns that exist in the DataFrame
        lag_columns = [col for col in lag_columns if col in result_df.columns]
    
    if lag_periods is None:
        lag_periods = [1, 2, 3, 4, 5, 6, 12, 24]
    
    if rolling_columns is None:
        rolling_columns = ['OT']
        # Filter to only include columns that exist in the DataFrame
        rolling_columns = [col for col in rolling_columns if col in result_df.columns]
    
    if rolling_windows is None:
        rolling_windows = [4, 12, 24, 48, 168]
    
    if fourier_periods is None:
        fourier_periods = {'daily': 24, 'weekly': 168, 'yearly': 8760}
    
    if fourier_order is None:
        fourier_order = {'daily': 4, 'weekly': 3, 'yearly': 5}
    
    # Apply feature engineering techniques
    if include_lag and lag_columns:
        # Check for NaN values in lag source columns
        lag_source_nans = result_df[lag_columns].isnull().sum()
        lag_cols_with_nans = lag_source_nans[lag_source_nans > 0]
        
        if not lag_cols_with_nans.empty:
            logger.warning(f"Source columns for lag features contain NaN values: \n{lag_cols_with_nans}")
            logger.info("Filling NaN values in lag source columns before creating lag features")
            
            for col in lag_cols_with_nans.index:
                # First try forward fill with a limit
                result_df[col] = result_df[col].ffill(limit=12)
                
                # Then use interpolation for remaining NaNs, but only for small gaps
                if result_df[col].isnull().sum() > 0:
                    # Find runs of NaNs
                    mask = result_df[col].isnull()
                    runs = mask.ne(mask.shift()).cumsum()
                    run_sizes = mask.groupby(runs).sum()
                    
                    # Find runs that are too large to interpolate
                    large_gaps = run_sizes[run_sizes > max_gap_limit].index
                    
                    if not large_gaps.empty:
                        # Get the indices of large gaps
                        large_gap_indices = mask[runs.isin(large_gaps)].index
                        logger.warning(f"Column '{col}' has {len(large_gaps)} gaps larger than {max_gap_limit} time steps")
                        
                        # Make a temporary copy of the series without large gaps for interpolation
                        temp_series = result_df[col].copy()
                        
                        # Interpolate the temporary series (ignoring large gaps)
                        temp_series = temp_series.interpolate(method='time')
                        
                        # Copy interpolated values back to original series, but only for small gaps
                        small_gap_mask = mask & ~runs.isin(large_gaps)
                        result_df.loc[small_gap_mask, col] = temp_series.loc[small_gap_mask]
                    else:
                        # No large gaps, just interpolate
                        result_df[col] = result_df[col].interpolate(method='time')
                
                # For any remaining missing values, use backward fill with limit
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].bfill(limit=12)
                
                # If there are still NaNs, fill with column median or mean
                if result_df[col].isnull().sum() > 0:
                    if result_df[col].notnull().sum() > 0:  # Make sure we have some non-NaN values
                        fill_value = result_df[col].median()
                        result_df[col] = result_df[col].fillna(fill_value)
                        logger.info(f"Filled remaining NaNs in column '{col}' with median: {fill_value}")
                    else:
                        # If all values are NaN, use 0 (but this is a serious issue)
                        logger.warning(f"Column '{col}' contains all NaNs, filling with 0")
                        result_df[col] = result_df[col].fillna(0)
        
        # Now create lag features
        result_df = create_lag_features(result_df, lag_columns, lag_periods)
        
        # Check for NaNs after creating lag features
        lag_nan_count = result_df.isnull().sum().sum()
        if lag_nan_count > initial_nan_count:
            logger.warning(f"Creating lag features introduced {lag_nan_count - initial_nan_count} new NaN values")
    
    # Similarly, ensure rolling feature source columns don't have NaNs
    if include_rolling and rolling_columns:
        # Check for NaN values in rolling source columns
        rolling_source_nans = result_df[rolling_columns].isnull().sum()
        rolling_cols_with_nans = rolling_source_nans[rolling_source_nans > 0]
        
        if not rolling_cols_with_nans.empty:
            logger.warning(f"Source columns for rolling features contain NaN values: \n{rolling_cols_with_nans}")
            logger.info("Filling NaN values in rolling source columns before creating rolling features")
            
            for col in rolling_cols_with_nans.index:
                # Similar approach to lag columns
                result_df[col] = result_df[col].ffill(limit=12)
                
                if result_df[col].isnull().sum() > 0:
                    # Interpolate but only for small gaps
                    mask = result_df[col].isnull()
                    runs = mask.ne(mask.shift()).cumsum()
                    run_sizes = mask.groupby(runs).sum()
                    large_gaps = run_sizes[run_sizes > max_gap_limit].index
                    
                    if not large_gaps.empty:
                        logger.warning(f"Column '{col}' has {len(large_gaps)} gaps larger than {max_gap_limit} time steps")
                        temp_series = result_df[col].copy()
                        temp_series = temp_series.interpolate(method='time')
                        small_gap_mask = mask & ~runs.isin(large_gaps)
                        result_df.loc[small_gap_mask, col] = temp_series.loc[small_gap_mask]
                    else:
                        result_df[col] = result_df[col].interpolate(method='time')
                
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].bfill(limit=12)
                
                if result_df[col].isnull().sum() > 0:
                    if result_df[col].notnull().sum() > 0:
                        fill_value = result_df[col].median()
                        result_df[col] = result_df[col].fillna(fill_value)
                        logger.info(f"Filled remaining NaNs in column '{col}' with median: {fill_value}")
                    else:
                        logger.warning(f"Column '{col}' contains all NaNs, filling with 0")
                        result_df[col] = result_df[col].fillna(0)
        
        # Now create rolling features
        result_df = create_rolling_features(result_df, rolling_columns, rolling_windows)
        
        # Check for NaNs after creating rolling features
        rolling_nan_count = result_df.isnull().sum().sum()
        if rolling_nan_count > lag_nan_count if include_lag else initial_nan_count:
            logger.warning(f"Creating rolling features introduced new NaN values")
    
    # For load ratio and load difference features, we need to ensure no NaN values in input columns
    required_load_columns = []
    if include_load_ratio or include_load_diff:
        required_load_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        available_load_columns = [col for col in required_load_columns if col in result_df.columns]
        
        if len(available_load_columns) < len(required_load_columns):
            missing_cols = set(required_load_columns) - set(available_load_columns)
            logger.warning(f"Missing load columns for ratio/difference features: {missing_cols}")
            logger.info(f"Will only use available columns: {available_load_columns}")
        
        # Check for NaN values in load columns
        load_nans = result_df[available_load_columns].isnull().sum()
        load_cols_with_nans = load_nans[load_nans > 0]
        
        if not load_cols_with_nans.empty:
            logger.warning(f"Load columns contain NaN values: \n{load_cols_with_nans}")
            logger.info("Filling NaN values in load columns before creating ratio/difference features")
            
            for col in load_cols_with_nans.index:
                # Similar approach to previous columns
                result_df[col] = result_df[col].ffill(limit=12)
                
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].interpolate(method='time')
                
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].bfill(limit=12)
                
                if result_df[col].isnull().sum() > 0:
                    if result_df[col].notnull().sum() > 0:
                        fill_value = result_df[col].median()
                        result_df[col] = result_df[col].fillna(fill_value)
                        logger.info(f"Filled remaining NaNs in column '{col}' with median: {fill_value}")
                    else:
                        logger.warning(f"Column '{col}' contains all NaNs, filling with 0")
                        result_df[col] = result_df[col].fillna(0)
    
    # Create load ratio features
    if include_load_ratio:
        result_df = create_load_ratio_features(result_df)
        
        # Check for NaNs after creating load ratio features
        ratio_nan_count = result_df.isnull().sum().sum()
        last_count = rolling_nan_count if include_rolling else (lag_nan_count if include_lag else initial_nan_count)
        if ratio_nan_count > last_count:
            logger.warning(f"Creating load ratio features introduced new NaN values")
    
    # Create load difference features
    if include_load_diff:
        result_df = create_load_difference_features(result_df)
        
        # Check for NaNs after creating load difference features
        diff_nan_count = result_df.isnull().sum().sum()
        last_count = ratio_nan_count if include_load_ratio else (rolling_nan_count if include_rolling else (lag_nan_count if include_lag else initial_nan_count))
        if diff_nan_count > last_count:
            logger.warning(f"Creating load difference features introduced new NaN values")
    
    # Add temperature rate features
    if include_temperature_rates and 'OT' in result_df.columns:
        logger.info("Adding temperature rate features")
        result_df = create_temperature_rate_features(
            result_df,
            temperature_column='OT',
            windows=[1, 2, 3, 6, 12, 24, 48]
        )
    
    # Add extreme temperature features
    if include_extreme_temp and 'OT' in result_df.columns:
        logger.info("Adding extreme temperature features")
        result_df = create_extreme_temperature_features(
            result_df,
            temperature_column='OT',
            threshold_quantile=0.9,
            windows=[24, 48, 72, 168]
        )
    
    # Add trend reversal features
    if include_trend_reversal:
        trend_columns = ['OT'] if 'OT' in result_df.columns else []
        
        # Add load columns if available
        load_columns = [col for col in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'] 
                        if col in result_df.columns]
        trend_columns.extend(load_columns)
        
        if trend_columns:
            logger.info(f"Adding trend reversal features for {len(trend_columns)} columns")
            result_df = create_trend_reversal_features(
                result_df,
                columns=trend_columns,
                windows=[6, 12, 24, 48]
            )
    
    # Add enhanced load features
    if include_enhanced_load:
        logger.info("Adding enhanced load features")
        result_df = create_enhanced_load_features(
            result_df,
            useful_load_columns=['HUFL', 'MUFL', 'LUFL'],
            useless_load_columns=['HULL', 'MULL', 'LULL'],
            windows=[4, 12, 24, 48]
        )
    
    # Use enhanced Fourier features if include_fourier is True
    if include_fourier:
        logger.info("Using enhanced Fourier features instead of standard ones")
        result_df = create_enhanced_fourier_features(
            result_df,
            periods=fourier_periods,
            fourier_orders={'daily': 6, 'weekly': 4, 'yearly': 8}  # Higher orders
        )
    
    # Handle missing values created during feature engineering with more advanced approach
    nan_count_before_final = result_df.isnull().sum().sum()
    if nan_count_before_final > 0:
        logger.info(f"Handling {nan_count_before_final} NaN values created during feature engineering")
        
        # First identify columns with NaN values
        columns_with_nans = result_df.columns[result_df.isnull().any()]
        logger.info(f"Found {len(columns_with_nans)} columns with NaN values")
        
        for col in columns_with_nans:
            nan_count_col = result_df[col].isnull().sum()
            
            # If only a small percentage of values are NaN, we can use more aggressive filling
            nan_percentage = nan_count_col / len(result_df) * 100
            
            if nan_percentage < 5:  # Less than 5% NaN values
                logger.info(f"Column '{col}' has {nan_percentage:.2f}% NaN values, applying aggressive filling")
                
                # Forward fill with larger limit
                result_df[col] = result_df[col].ffill(limit=24)
                
                # Then backward fill with larger limit
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].bfill(limit=24)
                
                # If still have NaNs, use median/mean
                if result_df[col].isnull().sum() > 0:
                    if result_df[col].notnull().sum() > 0:
                        fill_value = result_df[col].median()
                        result_df[col] = result_df[col].fillna(fill_value)
                        logger.info(f"Filled remaining NaNs in column '{col}' with median: {fill_value}")
            else:
                # For columns with higher percentage of NaNs, be more careful
                logger.info(f"Column '{col}' has {nan_percentage:.2f}% NaN values, applying careful filling")
                
                # First try forward fill with a limit
                result_df[col] = result_df[col].ffill(limit=12)
                
                # Then use interpolation for remaining NaNs, but only for small gaps
                if result_df[col].isnull().sum() > 0:
                    # Find runs of NaNs
                    mask = result_df[col].isnull()
                    runs = mask.ne(mask.shift()).cumsum()
                    run_sizes = mask.groupby(runs).sum()
                    
                    # Find runs that are too large to interpolate
                    large_gaps = run_sizes[run_sizes > max_gap_limit].index
                    
                    if not large_gaps.empty:
                        logger.warning(f"Column '{col}' has {len(large_gaps)} gaps larger than {max_gap_limit} time steps")
                        # Only interpolate small gaps
                        temp_series = result_df[col].copy()
                        temp_series = temp_series.interpolate(method='time')
                        small_gap_mask = mask & ~runs.isin(large_gaps)
                        result_df.loc[small_gap_mask, col] = temp_series.loc[small_gap_mask]
                    else:
                        # No large gaps, just interpolate
                        result_df[col] = result_df[col].interpolate(method='time')
                
                # For any remaining missing values, use backward fill with limit
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].bfill(limit=12)
        
        # If there are still missing values, report them
        nan_count_after = result_df.isnull().sum().sum()
        if nan_count_after > 0:
            logger.warning(f"After handling, still have {nan_count_after} NaN values")
            missing_by_col = result_df.isnull().sum()
            cols_with_missing = missing_by_col[missing_by_col > 0]
            logger.warning(f"Columns with remaining missing values: \n{cols_with_missing}")
            
            # For remaining NaNs in features, as a last resort, fill with median or 0
            for col in cols_with_missing.index:
                if result_df[col].notnull().sum() > 0:
                    fill_value = result_df[col].median()
                    result_df[col] = result_df[col].fillna(fill_value)
                    logger.info(f"Last resort: Filled remaining NaNs in column '{col}' with median: {fill_value}")
                else:
                    logger.warning(f"Column '{col}' contains all NaNs, filling with 0")
                    result_df[col] = result_df[col].fillna(0)
    
    # Final check
    final_nan_count = result_df.isnull().sum().sum()
    if final_nan_count > 0:
        logger.warning(f"Feature engineering complete but still have {final_nan_count} NaN values")
        missing_by_col = result_df.isnull().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0]
        logger.warning(f"Columns with missing values: \n{cols_with_missing}")
    else:
        logger.info("Feature engineering complete with no NaN values")
    
    logger.info(f"Feature set built, shape: {result_df.shape}")
    return result_df

def engineer_features(
    input_train_path: str,
    input_test_path: str,
    output_train_path: str,
    output_test_path: str,
    **feature_args
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to load preprocessed data, apply feature engineering, and save the results.
    
    Parameters
    ----------
    input_train_path : str
        Path to preprocessed training data
    input_test_path : str
        Path to preprocessed testing data
    output_train_path : str
        Path to save feature engineered training data
    output_test_path : str
        Path to save feature engineered testing data
    **feature_args : dict
        Additional arguments to pass to build_feature_set
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Feature engineered training and testing DataFrames
    """
    logger.info("Starting feature engineering process with enhanced features")
    
    # Load preprocessed data
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
    
    # Set enhanced feature engineering defaults if not provided
    if 'include_temperature_rates' not in feature_args:
        feature_args['include_temperature_rates'] = True
    if 'include_extreme_temp' not in feature_args:
        feature_args['include_extreme_temp'] = True
    if 'include_trend_reversal' not in feature_args:
        feature_args['include_trend_reversal'] = True
    if 'include_enhanced_load' not in feature_args:
        feature_args['include_enhanced_load'] = True
    
    # Build feature sets
    logger.info("Engineering enhanced features for training data")
    train_features = build_feature_set(train_df, **feature_args)
    
    logger.info("Engineering enhanced features for testing data")
    test_features = build_feature_set(test_df, **feature_args)
    
    # Save feature engineered data
    logger.info(f"Saving feature engineered training data to {output_train_path}")
    output_dir = os.path.dirname(output_train_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    try:
        train_features.to_csv(output_train_path)
        logger.info(f"Successfully saved training data with shape {train_features.shape}")
        
        # Save feature list for reference
        feature_list_path = os.path.join(output_dir, "feature_list.json")
        with open(feature_list_path, 'w') as f:
            feature_dict = {
                'all_features': train_features.columns.tolist(),
                'original_features': train_df.columns.tolist(),
                'engineered_features': [col for col in train_features.columns if col not in train_df.columns],
                'feature_counts': {
                    'total': len(train_features.columns),
                    'original': len(train_df.columns),
                    'engineered': len(train_features.columns) - len(train_df.columns)
                }
            }
            json.dump(feature_dict, f, indent=4)
        logger.info(f"Feature list saved to {feature_list_path}")
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
    
    # Create a feature importance analysis based on correlation with target
    if 'OT' in train_features.columns:
        logger.info("Analyzing feature importance based on correlation with target")
        
        # Calculate correlation with target
        corr_with_target = train_features.corrwith(train_features['OT']).abs().sort_values(ascending=False)
        
        # Save top features by correlation
        corr_path = os.path.join(output_dir, "feature_importance.json")
        with open(corr_path, 'w') as f:
            # Convert to dictionary with string keys and float values
            corr_dict = {
                'top_features': {str(k): float(v) for k, v in corr_with_target.head(30).items()},
                'bottom_features': {str(k): float(v) for k, v in corr_with_target.tail(30).items()},
                'feature_types': {
                    'lag_features': [col for col in train_features.columns if 'lag' in col],
                    'rolling_features': [col for col in train_features.columns if 'rolling' in col],
                    'fourier_features': [col for col in train_features.columns if 'sin' in col or 'cos' in col],
                    'trend_features': [col for col in train_features.columns if 'trend' in col],
                    'extreme_features': [col for col in train_features.columns if 'extreme' in col],
                    'rate_features': [col for col in train_features.columns if 'rate' in col or 'roc' in col]
                }
            }
            json.dump(corr_dict, f, indent=4)
        logger.info(f"Feature importance analysis saved to {corr_path}")
    
    logger.info("Enhanced feature engineering process completed")
    return train_features, test_features

if __name__ == "__main__":
    # Example usage
    project_dir = Path(__file__).resolve().parents[2]
    
    # Print the detected project directory
    print(f"Project directory: {project_dir}")
    
    # Input/output paths
    input_train_path = project_dir / "data" / "preprocessed" / "train_processed.csv"
    input_test_path = project_dir / "data" / "preprocessed" / "test_processed.csv"
    output_train_path = project_dir / "data" / "features" / "train_features_enhanced.csv"
    output_test_path = project_dir / "data" / "features" / "test_features_enhanced.csv"
    
    # Create output directory if it doesn't exist
    output_dir = project_dir / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input train path: {input_train_path}")
    print(f"Input test path: {input_test_path}")
    print(f"Output train path: {output_train_path}")
    print(f"Output test path: {output_test_path}")
    
    # Apply enhanced feature engineering
    train_features, test_features = engineer_features(
        input_train_path=str(input_train_path),
        input_test_path=str(input_test_path),
        output_train_path=str(output_train_path),
        output_test_path=str(output_test_path),
        include_lag=True,
        lag_columns=['OT', 'HUFL', 'MUFL', 'LUFL'],
        lag_periods=[1, 2, 3, 4, 6, 12, 24, 48],  # Added 48-hour lag
        include_rolling=True,
        rolling_columns=['OT'],
        rolling_windows=[4, 12, 24, 48, 168],
        include_load_ratio=True,
        include_load_diff=True,
        include_fourier=True,
        include_temperature_rates=True,
        include_extreme_temp=True,
        include_trend_reversal=True,
        include_enhanced_load=True
    )
    
    # Print summary of generated features
    print(f"Generated {train_features.shape[1]} features for training data")
    
    # Group features by type and count
    feature_counts = {
        'original': len([col for col in train_features.columns if col in ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]),
        'lag': len([col for col in train_features.columns if 'lag' in col]),
        'rolling': len([col for col in train_features.columns if 'rolling' in col]),
        'fourier': len([col for col in train_features.columns if 'sin' in col or 'cos' in col]),
        'load_ratio': len([col for col in train_features.columns if 'ratio' in col]),
        'load_diff': len([col for col in train_features.columns if 'diff' in col and 'temperature' not in col]),
        'temp_rate': len([col for col in train_features.columns if 'roc' in col or 'accel' in col or 'OT_diff' in col]),
        'extreme': len([col for col in train_features.columns if 'extreme' in col or 'hours_since_extreme' in col]),
        'trend': len([col for col in train_features.columns if 'trend' in col or 'oscillation' in col]),
        'enhanced_load': len([col for col in train_features.columns if 'volatility' in col or 'momentum' in col or 'temp_corr' in col])
    }
    
    print("Feature counts by type:")
    for feat_type, count in feature_counts.items():
        print(f"  {feat_type}: {count}")