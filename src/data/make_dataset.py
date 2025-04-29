"""
Functions for loading, preprocessing, and saving transformer oil temperature data.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Union
import json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded data
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    """
    logger.info(f"Loading data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_data(df: pd.DataFrame, output_path: str, index: bool = False) -> None:
    """
    Save DataFrame to a CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    output_path : str
        Path where the CSV file will be saved
    index : bool, optional
        Whether to save the DataFrame index, by default False
        
    Returns
    -------
    None
    """
    logger.info(f"Saving data to {output_path}")
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    try:
        df.to_csv(output_path, index=index)
        logger.info(f"Successfully saved data with shape {df.shape} to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def preprocess_data(
    df: pd.DataFrame,
    convert_date: bool = True,
    handle_missing: bool = True,
    handle_outliers: bool = True,
    resample_freq: Optional[str] = None,
    max_gap_limit: int = 24  # New parameter for maximum gap size to fill
) -> pd.DataFrame:
    """
    Preprocess transformer oil temperature data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    convert_date : bool, optional
        Whether to convert date column to datetime, by default True
    handle_missing : bool, optional
        Whether to handle missing values, by default True
    handle_outliers : bool, optional
        Whether to handle outliers, by default True
    resample_freq : str, optional
        Frequency for resampling, e.g., '15T' for 15 minutes, by default None
    max_gap_limit : int, optional
        Maximum size of gaps (in time steps) to fill using interpolation, by default 24
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    logger.info("Starting data preprocessing")
    
    # Make a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    
    # Convert date column to datetime
    if convert_date and 'date' in processed_df.columns:
        logger.info("Converting date column to datetime")
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # Set date as index
        processed_df.set_index('date', inplace=True)
        logger.info("Set date column as index")
    
    # Check and report missing values
    missing_values = processed_df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"Missing values per column: \n{missing_values[missing_values > 0]}")
    
    # Handle missing values with improved approach
    if handle_missing:
        logger.info("Handling missing values")
        
        # Forward fill small gaps (up to 4 time steps)
        initial_missing = processed_df.isnull().sum().sum()
        processed_df = processed_df.ffill(limit=4)
        after_ffill_missing = processed_df.isnull().sum().sum()
        logger.info(f"Forward fill reduced missing values from {initial_missing} to {after_ffill_missing}")
        
        # If there are still missing values, use interpolation with max_gap_limit
        if processed_df.isnull().sum().sum() > 0:
            logger.info(f"Using time interpolation with max gap limit of {max_gap_limit}")
            
            # For each column, interpolate but only for gaps smaller than max_gap_limit
            for col in processed_df.columns:
                # Get series with NaN values
                series = processed_df[col]
                
                # If the series has NaNs
                if series.isnull().any():
                    # Find runs of NaNs
                    mask = series.isnull()
                    
                    # Calculate runs of consecutive NaNs
                    # This works by taking the difference of cumulative sum of mask changes
                    runs = mask.ne(mask.shift()).cumsum()
                    
                    # Group by runs and get the size of each run
                    run_sizes = mask.groupby(runs).sum()
                    
                    # Find runs that are too large to interpolate
                    large_gaps = run_sizes[run_sizes > max_gap_limit].index
                    
                    if not large_gaps.empty:
                        # Get the indices of large gaps
                        large_gap_indices = mask[runs.isin(large_gaps)].index
                        logger.warning(f"Column '{col}' has {len(large_gaps)} gaps larger than {max_gap_limit} time steps")
                        
                        # Make a temporary copy of the series without large gaps for interpolation
                        temp_series = series.copy()
                        
                        # Interpolate the temporary series (ignoring large gaps)
                        temp_series = temp_series.interpolate(method='time')
                        
                        # Copy interpolated values back to original series, but only for small gaps
                        small_gap_mask = mask & ~runs.isin(large_gaps)
                        processed_df.loc[small_gap_mask, col] = temp_series.loc[small_gap_mask]
                    else:
                        # No large gaps, just interpolate
                        processed_df[col] = series.interpolate(method='time')
            
            after_interp_missing = processed_df.isnull().sum().sum()
            logger.info(f"Interpolation reduced missing values from {after_ffill_missing} to {after_interp_missing}")
            
            # For any remaining missing values at the beginning, use backward fill
            if processed_df.isnull().sum().sum() > 0:
                processed_df = processed_df.bfill(limit=4)
                after_bfill_missing = processed_df.isnull().sum().sum()
                logger.info(f"Backward fill reduced missing values from {after_interp_missing} to {after_bfill_missing}")
                
                # Report any remaining missing values
                if after_bfill_missing > 0:
                    missing_by_col = processed_df.isnull().sum()
                    cols_with_missing = missing_by_col[missing_by_col > 0]
                    logger.warning(f"After all imputation steps, still have missing values: \n{cols_with_missing}")
    
    # Handle outliers
    if handle_outliers:
        logger.info("Detecting and handling outliers with improved method")
        for column in processed_df.select_dtypes(include=[np.number]).columns:
            # Skip columns with NaN values for outlier detection
            if processed_df[column].isnull().any():
                logger.warning(f"Column '{column}' has NaN values, skipping outlier detection")
                continue
                
            # Instead of using a simple z-score approach, use a more robust method
            # Calculate IQR (Interquartile Range)
            Q1 = processed_df[column].quantile(0.25)
            Q3 = processed_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define less aggressive bounds for extreme values in oil temperature
            if column == 'OT':  # Oil Temperature column
                # Use a more permissive threshold for the target variable
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                
                # Identify outliers
                outliers = (processed_df[column] < lower_bound) | (processed_df[column] > upper_bound)
                
                if outliers.sum() > 0:
                    logger.info(f"Found {outliers.sum()} outliers in oil temperature column '{column}'")
                    
                    # For extreme temperatures, instead of replacing with median,
                    # cap the values at the bounds to preserve the extreme nature
                    processed_df.loc[processed_df[column] < lower_bound, column] = lower_bound
                    processed_df.loc[processed_df[column] > upper_bound, column] = upper_bound
                    logger.info(f"Capped extreme values in '{column}' at bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            else:
                # For other columns, use standard IQR method but with 2.5 instead of 1.5 for flexibility
                lower_bound = Q1 - 2.5 * IQR
                upper_bound = Q3 + 2.5 * IQR
                
                # Identify outliers
                outliers = (processed_df[column] < lower_bound) | (processed_df[column] > upper_bound)
                
                if outliers.sum() > 0:
                    logger.info(f"Found {outliers.sum()} outliers in column '{column}'")
                    
                    # Replace outliers with column median
                    column_median = processed_df[column].median()
                    processed_df.loc[outliers, column] = column_median
                    logger.info(f"Replaced outliers in '{column}' with median value: {column_median}")
    
    # Resample time series if needed
    if resample_freq is not None:
        logger.info(f"Resampling data to {resample_freq} frequency")
        
        # Make sure we have a datetime index
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex for resampling")
        
        # Resample
        processed_df = processed_df.resample(resample_freq).mean()
        
        # Handle any missing values created by resampling
        if processed_df.isnull().sum().sum() > 0:
            processed_df = processed_df.interpolate(method='time')
            logger.info("Interpolated missing values after resampling")
    
    # Final check for NaN values
    final_nan_count = processed_df.isnull().sum().sum()
    if final_nan_count > 0:
        logger.warning(f"Preprocessing complete but still have {final_nan_count} NaN values")
        missing_by_col = processed_df.isnull().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0]
        logger.warning(f"Columns with missing values: \n{cols_with_missing}")
    else:
        logger.info("Preprocessing complete with no remaining NaN values")
        
    logger.info(f"Preprocessing complete, final shape: {processed_df.shape}")
    return processed_df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime index.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional time features
        
    Raises
    ------
    ValueError
        If the DataFrame does not have a DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info("Creating time-based features")
    df_with_features = df.copy()
    
    # Extract time components
    df_with_features['hour'] = df.index.hour
    df_with_features['day_of_week'] = df.index.dayofweek
    df_with_features['day_of_year'] = df.index.dayofyear
    df_with_features['month'] = df.index.month
    df_with_features['quarter'] = df.index.quarter
    
    # Create cyclical features for periodic time components
    for col, period in [('hour', 24), ('day_of_week', 7), ('day_of_year', 365), ('month', 12), ('quarter', 4)]:
        df_with_features[f'{col}_sin'] = np.sin(2 * np.pi * df_with_features[col] / period)
        df_with_features[f'{col}_cos'] = np.cos(2 * np.pi * df_with_features[col] / period)
    
    # Drop the original time component columns
    df_with_features.drop(['hour', 'day_of_week', 'day_of_year', 'month', 'quarter'], axis=1, inplace=True)
    
    logger.info(f"Created cyclical time features, new shape: {df_with_features.shape}")
    return df_with_features

def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    valid_size: Optional[float] = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Split time series data into train and test sets, optionally including a validation set.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    test_size : float, optional
        Proportion of data to use for testing, by default 0.2
    valid_size : float, optional
        Proportion of data to use for validation, by default None
        
    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Train, test, and optionally validation DataFrames
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Sort by date
    df = df.sort_index()
    
    # Calculate split indices
    n = len(df)
    test_idx = int(n * (1 - test_size))
    
    if valid_size is not None:
        valid_idx = int(n * (1 - test_size - valid_size))
        logger.info(f"Splitting data: train={valid_idx}, validation={test_idx-valid_idx}, test={n-test_idx}")
        
        train_df = df.iloc[:valid_idx]
        valid_df = df.iloc[valid_idx:test_idx]
        test_df = df.iloc[test_idx:]
        
        return train_df, valid_df, test_df
    else:
        logger.info(f"Splitting data: train={test_idx}, test={n-test_idx}")
        
        train_df = df.iloc[:test_idx]
        test_df = df.iloc[test_idx:]
        
        return train_df, test_df

def process_file(input_path: str, output_path: str) -> None:
    """
    Process a single data file and save the result.
    
    Parameters
    ----------
    input_path : str
        Path to raw input data
    output_path : str
        Path to save processed data
    
    Returns
    -------
    None
    """
    # Load data
    df = load_data(input_path)
    
    # Analyze temperature distribution before preprocessing
    if 'OT' in df.columns:
        logger.info("Analyzing temperature distribution in raw data")
        raw_temp_stats = analyze_temperature_distribution(df, target_column='OT')
        
        # Save temperature statistics
        stats_dir = os.path.dirname(output_path)
        stats_file = os.path.join(stats_dir, 'temperature_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(raw_temp_stats, f, indent=4)
        logger.info(f"Temperature statistics saved to {stats_file}")
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Analyze temperature distribution after preprocessing
    if 'OT' in processed_df.columns:
        logger.info("Analyzing temperature distribution after preprocessing")
        processed_temp_stats = analyze_temperature_distribution(processed_df, target_column='OT')
        
        # Log any significant changes in distribution
        if raw_temp_stats['p95'] != processed_temp_stats['p95']:
            logger.info(f"95th percentile changed from {raw_temp_stats['p95']:.2f} to {processed_temp_stats['p95']:.2f}")
        if raw_temp_stats['max'] != processed_temp_stats['max']:
            logger.info(f"Maximum temperature changed from {raw_temp_stats['max']:.2f} to {processed_temp_stats['max']:.2f}")
    
    # Create time features
    processed_df = create_time_features(processed_df)
    
    # Save processed data
    save_data(processed_df, output_path, index=True)
    
    logger.info(f"Processing completed for {input_path}")
    return processed_df

def analyze_temperature_distribution(df: pd.DataFrame, target_column: str = 'OT') -> Dict[str, float]:
    """
    Analyze temperature distribution to identify important thresholds for preprocessing.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing temperature data
    target_column : str, optional
        Name of the temperature column, by default 'OT'
        
    Returns
    -------
    Dict[str, float]
        Dictionary with thresholds and distribution statistics
    """
    if target_column not in df.columns:
        logger.warning(f"Target column '{target_column}' not found in DataFrame")
        return {}
    
    logger.info(f"Analyzing temperature distribution for column '{target_column}'")
    
    # Get temperature values
    temps = df[target_column].dropna()
    
    # Calculate statistics
    stats = {
        'min': float(temps.min()),
        'max': float(temps.max()),
        'mean': float(temps.mean()),
        'median': float(temps.median()),
        'std': float(temps.std()),
        'q1': float(temps.quantile(0.25)),
        'q3': float(temps.quantile(0.75)),
        # Extreme thresholds
        'p90': float(temps.quantile(0.90)),  # 90th percentile
        'p95': float(temps.quantile(0.95)),  # 95th percentile
        'p99': float(temps.quantile(0.99)),  # 99th percentile
        # Low thresholds
        'p10': float(temps.quantile(0.10)),  # 10th percentile
        'p5': float(temps.quantile(0.05)),   # 5th percentile
        'p1': float(temps.quantile(0.01))    # 1st percentile
    }
    
    # Calculate IQR and bounds
    stats['iqr'] = stats['q3'] - stats['q1']
    stats['lower_bound'] = stats['q1'] - 3.0 * stats['iqr']
    stats['upper_bound'] = stats['q3'] + 3.0 * stats['iqr']
    
    # Log important statistics
    logger.info(f"Temperature range: {stats['min']:.2f} to {stats['max']:.2f}")
    logger.info(f"Temperature mean: {stats['mean']:.2f}, median: {stats['median']:.2f}")
    logger.info(f"Extreme temperature threshold (p95): {stats['p95']:.2f}")
    
    return stats

def main(train_input_path: str, test_input_path: str, train_output_path: str, test_output_path: str) -> None:
    """
    Main function to load, preprocess, and save train and test datasets.
    
    Parameters
    ----------
    train_input_path : str
        Path to raw training data
    test_input_path : str
        Path to raw testing data
    train_output_path : str
        Path to save processed training data
    test_output_path : str
        Path to save processed testing data
        
    Returns
    -------
    None
    """
    # Process training data
    logger.info("Processing training data")
    process_file(train_input_path, train_output_path)
    
    # Process testing data
    logger.info("Processing testing data")
    process_file(test_input_path, test_output_path)
    
    logger.info("Data processing complete for both training and testing datasets")

if __name__ == "__main__":
    # Example usage
    project_dir = Path(__file__).resolve().parents[2]
    
    # Print the detected project directory to help with debugging
    print(f"Project directory: {project_dir}")
    
    # Function to find a file with flexible path handling
    def find_file(base_filename):
        # Try the standard path first
        file_path = project_dir / "data" / "raw" / base_filename
        
        if file_path.exists():
            return file_path
                
        # If still not found, ask for manual input
        print(f"Please enter the full path to your {base_filename} file:")
        manual_path = input()
        manual_file = Path(manual_path)
        if not manual_file.exists():
            raise FileNotFoundError(f"Could not find the file at: {manual_path}")
        return manual_file
    
    # Find input files
    train_input_file = find_file("train.csv")
    test_input_file = find_file("test.csv")
    
    # Create output directories if they don't exist
    output_dir = project_dir / "data" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output_file = output_dir / "train_processed.csv"
    test_output_file = output_dir / "test_processed.csv"
    
    print(f"Input train file: {train_input_file}")
    print(f"Input test file: {test_input_file}")
    print(f"Output train file: {train_output_file}")
    print(f"Output test file: {test_output_file}")
    
    main(
        train_input_path=str(train_input_file),
        test_input_path=str(test_input_file),
        train_output_path=str(train_output_file),
        test_output_path=str(test_output_file)
    )