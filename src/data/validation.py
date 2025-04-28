"""
Functions for validating transformer oil temperature data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Validate raw data for transformer oil temperature forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data to validate
        
    Returns
    -------
    Tuple[bool, Dict]
        (is_valid, validation_results) 
        where is_valid is True if the data passes all validation checks
        and validation_results is a dictionary with details of each check
    """
    logger.info("Starting data validation")
    
    validation_results = {}
    all_checks_passed = True
    
    # Check 1: Required columns exist
    required_columns = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_results['required_columns_exist'] = {
        'passed': len(missing_columns) == 0,
        'missing_columns': missing_columns
    }
    
    if len(missing_columns) > 0:
        logger.warning(f"Missing required columns: {missing_columns}")
        all_checks_passed = False
    
    # Check 2: Date column is convertible to datetime
    if 'date' in df.columns:
        try:
            pd.to_datetime(df['date'])
            validation_results['date_convertible'] = {
                'passed': True
            }
        except Exception as e:
            validation_results['date_convertible'] = {
                'passed': False,
                'error': str(e)
            }
            logger.warning(f"Date column is not convertible to datetime: {str(e)}")
            all_checks_passed = False
    
    # Check 3: No duplicate timestamps
    if 'date' in df.columns:
        try:
            date_series = pd.to_datetime(df['date'])
            duplicates = date_series.duplicated()
            validation_results['no_duplicate_timestamps'] = {
                'passed': not duplicates.any(),
                'num_duplicates': duplicates.sum()
            }
            
            if duplicates.any():
                logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
                all_checks_passed = False
                
        except Exception as e:
            validation_results['no_duplicate_timestamps'] = {
                'passed': False,
                'error': str(e)
            }
            logger.warning(f"Could not check for duplicate timestamps: {str(e)}")
            all_checks_passed = False
    
    # Check 4: Data is in expected range
    expected_ranges = {
        'HUFL': (-20, 50),
        'HULL': (-5, 20),
        'MUFL': (-20, 50),
        'MULL': (-5, 20),
        'LUFL': (-20, 50),
        'LULL': (-5, 20),
        'OT': (-5, 50)
    }
    
    range_validation = {}
    for col, (min_val, max_val) in expected_ranges.items():
        if col in df.columns:
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()
            
            range_validation[col] = {
                'passed': below_min == 0 and above_max == 0,
                'below_min': below_min,
                'above_max': above_max
            }
            
            if below_min > 0 or above_max > 0:
                logger.warning(f"Column '{col}' has {below_min} values below {min_val} and {above_max} values above {max_val}")
                all_checks_passed = False
    
    validation_results['data_in_range'] = range_validation
    
    # Check 5: Missing values
    missing_counts = df.isnull().sum()
    has_missing = missing_counts.sum() > 0
    
    validation_results['no_missing_values'] = {
        'passed': not has_missing,
        'missing_counts': missing_counts[missing_counts > 0].to_dict() if has_missing else {}
    }
    
    if has_missing:
        logger.warning(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
        # We don't fail validation for missing values, as they can be handled during preprocessing
    
    # Check 6: Consistent time intervals
    if 'date' in df.columns:
        try:
            date_series = pd.to_datetime(df['date']).sort_values()
            time_diffs = date_series.diff().dropna()
            
            # Convert to minutes
            time_diffs_minutes = time_diffs.dt.total_seconds() / 60
            
            # Get the most common interval
            most_common_interval = time_diffs_minutes.value_counts().idxmax()
            consistent_intervals = (time_diffs_minutes == most_common_interval).mean() >= 0.95
            
            validation_results['consistent_time_intervals'] = {
                'passed': consistent_intervals,
                'most_common_interval_minutes': most_common_interval,
                'consistency_percentage': (time_diffs_minutes == most_common_interval).mean() * 100
            }
            
            if not consistent_intervals:
                logger.warning(f"Time intervals are not consistent. Most common interval: {most_common_interval} minutes occurs in {(time_diffs_minutes == most_common_interval).mean() * 100:.2f}% of cases")
                all_checks_passed = False
                
        except Exception as e:
            validation_results['consistent_time_intervals'] = {
                'passed': False,
                'error': str(e)
            }
            logger.warning(f"Could not check time intervals: {str(e)}")
            all_checks_passed = False
    
    # Overall validation result
    logger.info(f"Data validation {'passed' if all_checks_passed else 'failed'}")
    
    return all_checks_passed, validation_results

def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for missing values in each column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to check
        
    Returns
    -------
    Dict[str, int]
        Dictionary with column names as keys and number of missing values as values
    """
    missing_values = df.isnull().sum().to_dict()
    return {col: count for col, count in missing_values.items() if count > 0}

def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zscore',
    threshold: float = 3.0
) -> Dict[str, pd.Series]:
    """
    Detect outliers in specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to check
    columns : List[str], optional
        List of columns to check, by default None (checks all numeric columns)
    method : str, optional
        Method to use for outlier detection, by default 'zscore'
        Options: 'zscore', 'iqr'
    threshold : float, optional
        Threshold for outlier detection, by default 3.0
        For zscore: values with absolute z-score > threshold are outliers
        For iqr: values outside threshold * IQR from quartiles are outliers
        
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary with column names as keys and boolean Series as values
        (True for outliers, False otherwise)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.warning(f"Column '{column}' is not numeric, skipping outlier detection")
            continue
        
        if method == 'zscore':
            # Z-score method
            mean = df[column].mean()
            std = df[column].std()
            
            # Avoid division by zero
            if std == 0:
                logger.warning(f"Column '{column}' has zero standard deviation, skipping outlier detection")
                continue
                
            z_scores = np.abs((df[column] - mean) / std)
            outliers[column] = z_scores > threshold
            
        elif method == 'iqr':
            # IQR method
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers[column] = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        num_outliers = outliers[column].sum()
        logger.info(f"Detected {num_outliers} outliers in column '{column}' using {method} method")
    
    return outliers

def validate_time_series_completeness(df: pd.DataFrame, freq: str = '15T') -> Tuple[bool, pd.DataFrame]:
    """
    Validate that a time series DataFrame has complete timestamps without gaps.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    freq : str, optional
        Expected frequency of the time series, by default '15T' (15 minutes)
        
    Returns
    -------
    Tuple[bool, pd.DataFrame]
        (is_complete, missing_periods)
        where is_complete is True if there are no gaps in the time series
        and missing_periods is a DataFrame with the missing time periods
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            if 'date' in df.columns:
                df = df.set_index('date')
                df.index = pd.to_datetime(df.index)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or 'date' column")
        except Exception as e:
            logger.error(f"Could not convert index to DatetimeIndex: {str(e)}")
            raise
    
    # Create a complete date range
    complete_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    
    # Find missing dates
    missing_dates = complete_range.difference(df.index)
    
    is_complete = len(missing_dates) == 0
    
    if not is_complete:
        logger.warning(f"Found {len(missing_dates)} missing timestamps in time series")
        
        # Create a DataFrame of missing periods
        missing_periods = pd.DataFrame(index=missing_dates)
        missing_periods['is_missing'] = True
        
        return is_complete, missing_periods
    else:
        logger.info("Time series is complete with no missing timestamps")
        return is_complete, pd.DataFrame()

def identify_sudden_changes(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold_multiplier: float = 3.0
) -> Dict[str, pd.Series]:
    """
    Identify sudden changes in time series data that may indicate anomalies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data
    columns : List[str], optional
        List of columns to check, by default None (checks all numeric columns)
    threshold_multiplier : float, optional
        Multiplier for the standard deviation of differences to identify sudden changes,
        by default 3.0
        
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary with column names as keys and boolean Series as values
        (True for sudden changes, False otherwise)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            if 'date' in df.columns:
                df = df.set_index('date')
                df.index = pd.to_datetime(df.index)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or 'date' column")
        except Exception as e:
            logger.error(f"Could not convert index to DatetimeIndex: {str(e)}")
            raise
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    sudden_changes = {}
    
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.warning(f"Column '{column}' is not numeric, skipping")
            continue
        
        # Calculate first differences
        diff = df[column].diff()
        
        # Calculate mean and standard deviation of differences
        mean_diff = diff.mean()
        std_diff = diff.std()
        
        # Identify sudden changes
        threshold = threshold_multiplier * std_diff
        sudden_changes[column] = (diff > mean_diff + threshold) | (diff < mean_diff - threshold)
        
        num_changes = sudden_changes[column].sum()
        logger.info(f"Detected {num_changes} sudden changes in column '{column}'")
    
    return sudden_changes

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from pathlib import Path
    
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir / "data/raw/train.csv"
    
    # Load sample data
    df = pd.read_csv(input_file)
    
    # Validate the data
    is_valid, validation_results = validate_raw_data(df)
    
    print(f"Data validation {'passed' if is_valid else 'failed'}")
    print(f"Validation results: {validation_results}")