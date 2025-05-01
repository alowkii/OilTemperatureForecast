"""
Functions for loading models and making predictions for oil temperature forecasting.

This module contains functions to:
1. Load trained models and associated artifacts
2. Prepare data for prediction
3. Make predictions on new data
4. Denormalize predictions to original scale
"""

import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_model(
    model_path: str,
    scaler_path: Optional[str] = None,
    history_path: Optional[str] = None
) -> Tuple[tf.keras.Model, Optional[Any], Optional[Dict]]:
    """
    Load trained model and associated artifacts.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
    scaler_path : str, optional
        Path to the saved scaler, by default None
    history_path : str, optional
        Path to the saved training history, by default None
        If None, will try to infer from model_path
        
    Returns
    -------
    Tuple[tf.keras.Model, Optional[Any], Optional[Dict]]
        Loaded model, scaler (if available), and training history (if available)
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf_load_model(model_path)
    
    # Load scaler if path is provided
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Try to infer history path if not provided
    if history_path is None:
        history_path = os.path.splitext(model_path)[0] + '_history.json'
    
    # Load training history if available
    history = None
    if os.path.exists(history_path):
        logger.info(f"Loading training history from {history_path}")
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    return model, scaler, history

def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features that might be needed for prediction but are missing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional basic features
    """
    logger.info("Creating basic features for missing columns")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # Dictionary to store new features
    new_features = {}
    
    # Create lag features of OT if OT exists but lag features don't
    if 'OT' in df_new.columns:
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            lag_col = f'OT_lag_{lag}'
            if lag_col not in df_new.columns:
                new_features[lag_col] = df_new['OT'].shift(lag)
    
    # Create lag features for main load columns if they exist
    for col in ['HUFL', 'MUFL', 'LUFL']:
        if col in df_new.columns:
            for lag in [1, 24, 48]:
                lag_col = f'{col}_lag_{lag}'
                if lag_col not in df_new.columns:
                    new_features[lag_col] = df_new[col].shift(lag)
    
    # Create basic time features
    if isinstance(df_new.index, pd.DatetimeIndex):
        # Hour of day
        hour = df_new.index.hour
        
        # Day of week
        day_of_week = df_new.index.dayofweek
        
        # Month
        month = df_new.index.month
        
        # Basic cyclical encoding for time
        if 'daily_sin_1' not in df_new.columns:
            new_features['daily_sin_1'] = np.sin(2 * np.pi * hour / 24)
            new_features['daily_cos_1'] = np.cos(2 * np.pi * hour / 24)
        
        if 'weekly_sin_1' not in df_new.columns:
            new_features['weekly_sin_1'] = np.sin(2 * np.pi * day_of_week / 7)
            new_features['weekly_cos_1'] = np.cos(2 * np.pi * day_of_week / 7)
        
        if 'yearly_sin_1' not in df_new.columns:
            new_features['yearly_sin_1'] = np.sin(2 * np.pi * month / 12)
            new_features['yearly_cos_1'] = np.cos(2 * np.pi * month / 12)
    
    # Create DataFrame with new features
    if new_features:
        new_features_df = pd.DataFrame(new_features, index=df_new.index)
        
        # Concatenate with original data
        result = pd.concat([df_new, new_features_df], axis=1)
        
        logger.info(f"Created {len(new_features)} basic features")
        return result
    
    return df_new

def prepare_data_for_prediction(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str],
    scaler: Optional[Any] = None,
    handle_nans: bool = True
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Prepare data for making predictions with LSTM model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features, should have datetime index
    sequence_length : int
        Length of input sequence for the model
    feature_columns : List[str]
        List of feature columns to use
    scaler : Any, optional
        Fitted scaler for normalization, by default None
    handle_nans : bool, optional
        Whether to handle NaN values before creating sequences, by default True
        
    Returns
    -------
    Tuple[np.ndarray, pd.DatetimeIndex]
        Input sequences ready for prediction and the corresponding timestamps
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Preparing data for prediction with sequence length {sequence_length}")
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Ensure data is sorted by index
    data = data.sort_index()
    
    # Check for required feature columns
    available_columns = data.columns.tolist()
    missing_columns = [col for col in feature_columns if col not in available_columns]
    
    if missing_columns:
        logger.warning(f"Missing {len(missing_columns)} feature columns required by the model")
        logger.warning(f"First few missing features: {missing_columns[:5]}")
        
        # Check if we should use available features only or attempt to create missing features
        # For this fix, we'll use only available features
        feature_columns = [col for col in feature_columns if col in available_columns]
        
        if not feature_columns:
            raise ValueError("No required features are available in the dataset")
        
        logger.info(f"Proceeding with {len(feature_columns)} available features")
    
    # Check for NaN values in feature columns
    nan_count = data[feature_columns].isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in feature columns")
        
        if handle_nans:
            logger.info("Handling NaN values in features")
            # First try forward fill with a limit
            data[feature_columns] = data[feature_columns].ffill(limit=12)
            
            # Then use interpolation for remaining NaNs
            if data[feature_columns].isna().sum().sum() > 0:
                data[feature_columns] = data[feature_columns].interpolate(method='time')
            
            # Finally use backward fill for any remaining NaNs (usually at the beginning)
            if data[feature_columns].isna().sum().sum() > 0:
                data[feature_columns] = data[feature_columns].bfill()
            
            # Check if we still have NaNs
            remaining_nans = data[feature_columns].isna().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"Still have {remaining_nans} NaN values after handling")
    
    # Scale data if scaler is provided
    if scaler:
        logger.info("Scaling data")
        
        # Check if all columns required by scaler are present
        if hasattr(scaler, 'feature_names_in_'):
            scaler_columns = scaler.feature_names_in_
            
            # If we have column name information in the scaler
            columns_for_scaling = [col for col in scaler_columns if col in data.columns]
            
            if len(columns_for_scaling) < len(scaler_columns):
                logger.warning(f"Missing {len(scaler_columns) - len(columns_for_scaling)} columns required by scaler")
                # Create dummy columns with zeros for missing features
                missing_scaler_cols = [col for col in scaler_columns if col not in data.columns]
                missing_df = pd.DataFrame(0.0, index=data.index, columns=missing_scaler_cols)
                
                # Concatenate with original data
                data = pd.concat([data, missing_df], axis=1)
                logger.info(f"Created {len(missing_scaler_cols)} dummy columns for scaling")
            
            # Now scale with all required columns
            scaled_values = scaler.transform(data[scaler_columns])
            # Create a new DataFrame with scaled values
            scaled_df = pd.DataFrame(scaled_values, index=data.index, columns=scaler_columns)
            # Replace only the columns we need for prediction
            for col in feature_columns:
                if col in scaled_df.columns:
                    data[col] = scaled_df[col]
        else:
            # No column information, try standard scaling
            scaled_values = scaler.transform(data)
            data = pd.DataFrame(scaled_values, index=data.index, columns=data.columns)
    
    # Extract feature columns
    X_values = data[feature_columns].values
    
    # Create sequences
    X_sequences = []
    timestamps = []
    
    # For prediction, we create sequences but no targets (since we're predicting them)
    for i in range(len(data) - sequence_length + 1):
        seq = X_values[i:i+sequence_length]
        
        # Skip sequences with NaN values
        if np.isnan(seq).any():
            logger.debug(f"Skipping sequence at position {i} due to NaN values")
            continue
            
        X_sequences.append(seq)
        # Store the timestamp of the last point in the input sequence
        timestamps.append(data.index[i+sequence_length-1])
    
    logger.info(f"Created {len(X_sequences)} prediction sequences")
    
    if len(X_sequences) == 0:
        logger.warning("No valid sequences could be created due to NaN values")
        # Return empty arrays with correct shapes
        return np.empty((0, sequence_length, len(feature_columns))), pd.DatetimeIndex([])
    
    return np.array(X_sequences), pd.DatetimeIndex(timestamps)

def make_predictions(
    model: tf.keras.Model,
    X: np.ndarray,
    scaler: Optional[Any] = None,
    target_column_idx: Optional[int] = None,
    inverse_transform_params: Optional[Dict] = None
) -> np.ndarray:
    """
    Make predictions with the model and denormalize if needed.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model
    X : np.ndarray
        Input sequences
    scaler : Any, optional
        Fitted scaler for denormalization, by default None
    target_column_idx : int, optional
        Index of target column in original data (for denormalization), by default None
    inverse_transform_params : Dict, optional
        Additional parameters for inverse transformation, by default None
        
    Returns
    -------
    np.ndarray
        Model predictions, denormalized if scaler was provided
    """
    logger.info(f"Making predictions for {len(X)} sequences")
    
    # Check if the model requires two inputs (encoder-decoder with attention)
    if isinstance(model.input, list) and len(model.input) == 2:
        logger.info("Detected encoder-decoder model with two inputs (encoder + decoder)")
        
        # For encoder-decoder models with attention, we need to provide the decoder input
        # Create a dummy decoder input with zeros (for inference)
        decoder_input = np.zeros((len(X), 1, 1))
        
        # Make predictions
        predictions = model.predict([X, decoder_input])
    else:
        # For single-input models
        predictions = model.predict(X)
    
    # Denormalize predictions if scaler is provided
    if scaler and target_column_idx is not None:
        logger.info("Denormalizing predictions")
        
        # Get prediction shape and handle different model output formats
        pred_shape = predictions.shape
        logger.info(f"Raw prediction shape: {pred_shape}")
        
        # Handle the specific shape (samples, 1, time_steps) from encoder-decoder
        if len(pred_shape) == 3:
            if pred_shape[1] == 1 and pred_shape[2] > 1:
                # Shape is (samples, 1, time_steps) - need to reshape to (samples, time_steps)
                logger.info(f"Reshaping from {pred_shape} to (samples, time_steps)")
                predictions = predictions.squeeze(axis=1)
                logger.info(f"New shape after squeeze: {predictions.shape}")
            elif pred_shape[2] == 1 and pred_shape[1] > 1:
                # Shape is (samples, time_steps, 1) - need to reshape to (samples, time_steps)
                logger.info(f"Reshaping from {pred_shape} to (samples, time_steps)")
                predictions = predictions.squeeze(axis=2)
                logger.info(f"New shape after squeeze: {predictions.shape}")
        
        # Now proceed with denormalization
        n_samples = predictions.shape[0]
        n_steps = predictions.shape[1]
        
        logger.info(f"Denormalizing {n_samples} samples with {n_steps} steps each")
        
        # Create empty array for denormalized predictions
        denormalized_predictions = np.zeros((n_samples, n_steps))
        
        # Process each sample and step individually to avoid broadcasting issues
        for i in range(n_samples):
            for j in range(n_steps):
                # Create a dummy array with all zeros except for the target value
                dummy = np.zeros(scaler.n_features_in_)
                dummy[target_column_idx] = predictions[i, j]
                
                # Inverse transform this single point
                denorm_point = scaler.inverse_transform(dummy.reshape(1, -1))
                
                # Store the denormalized value
                denormalized_predictions[i, j] = denorm_point[0, target_column_idx]
        
        logger.info(f"Denormalized predictions shape: {denormalized_predictions.shape}")
        return denormalized_predictions
    
    # If no scaler or target_column_idx provided, return raw predictions
    return predictions

def forecast_future(
    model: tf.keras.Model,
    last_sequence: np.ndarray,
    forecast_horizon: int,
    scaler: Optional[Any] = None,
    target_column_idx: Optional[int] = None,
    feature_columns: Optional[List[str]] = None
) -> np.ndarray:
    """
    Make multi-step forecasts by iteratively using model predictions as inputs for future steps.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model
    last_sequence : np.ndarray
        Last known input sequence, shape (1, sequence_length, n_features)
    forecast_horizon : int
        Number of future steps to forecast
    scaler : Any, optional
        Fitted scaler for denormalization, by default None
    target_column_idx : int, optional
        Index of target column in features (for denormalization), by default None
    feature_columns : List[str], optional
        Names of feature columns (for logging), by default None
        
    Returns
    -------
    np.ndarray
        Multi-step forecast, shape (1, forecast_horizon)
    """
    logger.info(f"Forecasting {forecast_horizon} steps into the future")
    
    # Make a copy of the last sequence
    sequence = last_sequence.copy()
    
    # Check input shape
    if len(sequence.shape) != 3:
        raise ValueError(f"Expected 3D input (1, sequence_length, n_features), got shape {sequence.shape}")
    
    # Store predictions
    predictions = np.zeros((1, forecast_horizon))
    
    # Check if the model requires two inputs (encoder-decoder with attention)
    is_encoder_decoder = isinstance(model.input, list) and len(model.input) == 2
    
    for i in range(forecast_horizon):
        # Predict the next step
        if is_encoder_decoder:
            # Create dummy decoder input (single zero timestep)
            decoder_input = np.zeros((1, 1, 1))
            next_pred = model.predict([sequence, decoder_input])
        else:
            next_pred = model.predict(sequence)
        
        # Handle different prediction shapes
        if len(next_pred.shape) > 2:
            # If prediction is 3D (batch, time_steps, features)
            # Take the first time step and first feature
            next_val = next_pred[0, 0, 0]
        elif len(next_pred.shape) == 2:
            # If prediction is 2D (batch, time_steps)
            # Take the first time step
            next_val = next_pred[0, 0]
        else:
            # For unusual shapes, just take the first element
            next_val = next_pred[0]
            
        # Store the prediction for this step
        predictions[0, i] = next_val
        
        # Update sequence for next iteration by shifting and adding new prediction
        # Remove the oldest time step
        sequence = sequence[:, 1:, :]
        
        # Create new step based on the last time step
        new_step = sequence[:, -1, :].copy()
        
        # Update the target feature with our prediction
        if target_column_idx is not None:
            new_step[0, target_column_idx] = next_val
        
        # Add the new step to the sequence
        sequence = np.concatenate([sequence, new_step.reshape(1, 1, -1)], axis=1)
    
    # Denormalize predictions if scaler is provided
    if scaler and target_column_idx is not None:
        logger.info("Denormalizing predictions")
        
        # Create a dummy array with correct shape for inverse_transform
        dummy = np.zeros((forecast_horizon, scaler.n_features_in_))
        dummy[:, target_column_idx] = predictions[0, :]
        
        # Inverse transform
        denorm_predictions = scaler.inverse_transform(dummy)
        
        # Extract just the target column
        return denorm_predictions[:, target_column_idx].reshape(1, -1)
    
    return predictions

def generate_future_timestamps(
    start_timestamp: pd.Timestamp,
    forecast_horizon: int,
    freq: str = 'h'
) -> pd.DatetimeIndex:
    """
    Generate future timestamps for forecasts.
    
    Parameters
    ----------
    start_timestamp : pd.Timestamp
        Starting timestamp (last known time point)
    forecast_horizon : int
        Number of future timestamps to generate
    freq : str, optional
        Frequency of timestamps, by default 'h' (hourly)
        
    Returns
    -------
    pd.DatetimeIndex
        Future timestamps
    """
    # Convert frequency to timedelta for the first step
    if freq == 'h' or freq == 'H':
        time_delta = pd.Timedelta(hours=1)
    elif freq == 'd' or freq == 'D':
        time_delta = pd.Timedelta(days=1)
    elif freq == 'min' or freq == 'T':
        time_delta = pd.Timedelta(minutes=1)
    elif freq == '15min' or freq == '15T':
        time_delta = pd.Timedelta(minutes=15)
    elif freq == '30min' or freq == '30T':
        time_delta = pd.Timedelta(minutes=30)
    else:
        # Default case
        time_delta = pd.Timedelta(hours=1)
        
    # Generate future timestamps using date_range
    return pd.date_range(
        start=start_timestamp + time_delta,
        periods=forecast_horizon,
        freq=freq
    )

def predict_dataset(
    model: tf.keras.Model,
    df: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    feature_columns: List[str],
    target_column: str,
    scaler: Optional[Any] = None,
    freq: str = 'h',
    return_actual: bool = True,
    handle_nans: bool = True
) -> pd.DataFrame:
    """
    Make predictions for an entire dataset and return results as a DataFrame.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model
    df : pd.DataFrame
        Input DataFrame with features and optionally targets
    sequence_length : int
        Length of input sequences
    forecast_horizon : int
        Number of future steps to predict
    feature_columns : List[str]
        List of feature column names
    target_column : str
        Name of target column
    scaler : Any, optional
        Fitted scaler, by default None
    freq : str, optional
        Time series frequency, by default 'h' (hourly)
    return_actual : bool, optional
        Whether to include actual values in output, by default True
    handle_nans : bool, optional
        Whether to handle NaN values before prediction, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and optionally actual values
    """
    logger.info(f"Predicting for dataset with {len(df)} rows")
    
    # Create a deep copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Check for required feature columns
    available_columns = data.columns.tolist()
    missing_columns = [col for col in feature_columns if col not in available_columns]
    
    if missing_columns:
        logger.warning(f"Missing {len(missing_columns)} feature columns required by the model")
        logger.info(f"First few missing features: {missing_columns[:10]}")
        
        # Try to create basic missing features
        data = create_basic_features(data)
        
        # Recheck missing columns after feature creation
        available_columns = data.columns.tolist()
        still_missing = [col for col in feature_columns if col not in available_columns]
        
        if still_missing:
            logger.warning(f"Still missing {len(still_missing)} features after attempting to create basic features")
            
            # Create dummy columns with zeros for remaining missing features
            # Instead of adding one by one (which causes DataFrame fragmentation),
            # create a DataFrame with all missing columns and then concatenate
            missing_df = pd.DataFrame(0.0, 
                                     index=data.index, 
                                     columns=still_missing)
            
            # Concatenate with original data
            data = pd.concat([data, missing_df], axis=1)
            
            logger.info(f"Created {len(still_missing)} dummy columns filled with zeros")
    
    # Handle NaNs in input data if requested
    if handle_nans:
        # Check for NaN values in feature columns
        nan_count = data[feature_columns].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in feature columns")
            logger.info("Handling NaN values in features")
            
            # First try forward fill with a limit
            data[feature_columns] = data[feature_columns].ffill(limit=12)
            
            # Then use interpolation for remaining NaNs
            if data[feature_columns].isna().sum().sum() > 0:
                data[feature_columns] = data[feature_columns].interpolate(method='time')
            
            # Finally use backward fill for any remaining NaNs (usually at the beginning)
            if data[feature_columns].isna().sum().sum() > 0:
                data[feature_columns] = data[feature_columns].bfill()
            
            # Check if we still have NaNs
            remaining_nans = data[feature_columns].isna().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"Still have {remaining_nans} NaN values after handling")
                # Fill remaining NaNs with 0
                data[feature_columns] = data[feature_columns].fillna(0)
    
    # Prepare data for prediction
    X, timestamps = prepare_data_for_prediction(
        data, sequence_length, feature_columns, scaler, handle_nans=False  # NaNs already handled
    )
    
    # Check if we have any valid sequences
    if len(X) == 0:
        logger.warning("No valid sequences could be created due to NaN values")
        return pd.DataFrame()
    
    # Get target column index for denormalization
    target_idx = None
    if scaler:
        if hasattr(scaler, 'feature_names_in_'):
            # Get index from scaler's feature names
            scaler_columns = list(scaler.feature_names_in_)
            if target_column in scaler_columns:
                target_idx = scaler_columns.index(target_column)
                logger.info(f"Found target column '{target_column}' at index {target_idx} in scaler features")
            else:
                logger.warning(f"Target column '{target_column}' not found in scaler's feature names")
        else:
            # Try to get index from dataframe columns
            if target_column in data.columns:
                target_idx = list(data.columns).index(target_column)
                logger.info(f"Using target column '{target_column}' at index {target_idx} from DataFrame columns")
            else:
                logger.warning(f"Target column '{target_column}' not found in DataFrame")
    
    # Make predictions
    logger.info(f"Calling make_predictions with X shape {X.shape}")
    predictions = make_predictions(model, X, scaler, target_idx)
    logger.info(f"Received predictions with shape {predictions.shape}")
    
    # Create result DataFrame
    result_dfs = []
    
    # For each prediction (which is multi-step)
    for i, timestamp in enumerate(timestamps):
        # Generate forecast timestamps
        forecast_times = generate_future_timestamps(
            timestamp, forecast_horizon, freq
        )
        
        # Ensure we have the right shape for each prediction row
        # Handle case where prediction shape might have been squeezed
        if len(predictions.shape) == 1:
            row_prediction = predictions.reshape(1, -1)[0]
        else:
            row_prediction = predictions[i]
        
        # Ensure we're only using forecast_horizon steps even if model outputs more
        if len(row_prediction) > forecast_horizon:
            logger.warning(f"Model predicted {len(row_prediction)} steps, but we only requested {forecast_horizon}")
            row_prediction = row_prediction[:forecast_horizon]
        elif len(row_prediction) < forecast_horizon:
            logger.warning(f"Model predicted only {len(row_prediction)} steps, but we need {forecast_horizon}")
            # Pad with the last value if necessary
            padding = np.full(forecast_horizon - len(row_prediction), row_prediction[-1])
            row_prediction = np.concatenate([row_prediction, padding])
            
        # Create DataFrame for this forecast
        forecast_df = pd.DataFrame(
            row_prediction.reshape(forecast_horizon, 1),
            index=forecast_times,
            columns=['predicted']
        )
        
        # Add actual values if requested and available
        if return_actual and target_column in data.columns:
            # Check if we have actual data for the forecast period
            # Safely get actual data that exists in both forecast_times and df.index
            actual_indices = forecast_times[forecast_times.isin(data.index)]
            
            if len(actual_indices) > 0:
                # Get values for indices that exist
                actual_values = data.loc[actual_indices, target_column]
                
                # Create a Series with forecast_times as index, filled with NaN
                all_actuals = pd.Series(index=forecast_times, dtype=float)
                
                # Fill in available values
                all_actuals.loc[actual_indices] = actual_values
                
                # Add to forecast DataFrame
                forecast_df['actual'] = all_actuals.values
        
        # Add sequence ID
        forecast_df['sequence_id'] = i
        
        # Add prediction start time
        forecast_df['prediction_start'] = timestamp
        
        result_dfs.append(forecast_df)
    
    # Combine all forecasts
    if result_dfs:
        result = pd.concat(result_dfs)
        logger.info(f"Created prediction DataFrame with {len(result)} rows")
        return result
    else:
        logger.warning("No predictions generated")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    project_dir = Path(__file__).resolve().parents[2]
    
    # Print the detected project directory
    print(f"Project directory: {project_dir}")
    
    # Define paths
    test_features_path = project_dir / "data" / "features" / "test_features.csv"
    model_dir = project_dir / "models"
    model_path = model_dir / "lstm_encoder_decoder_model.keras"  # Updated model path
    scaler_path = model_dir / "scaler.pkl"
    predictions_path = project_dir / "data" / "predictions" / "test_predictions.csv"
    
    # Create predictions directory if needed
    predictions_dir = project_dir / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and artifacts
    model, scaler, history = load_model(
        model_path=str(model_path),
        scaler_path=str(scaler_path)
    )
    
    # Check if the model requires two inputs (encoder-decoder with attention)
    is_encoder_decoder = isinstance(model.input, list) and len(model.input) == 2
    if is_encoder_decoder:
        print(f"Detected encoder-decoder model with attention mechanism")
    else:
        print(f"Detected single-input model architecture")
    
    # Get metadata from history
    if history and 'metadata' in history:
        metadata = history['metadata']
        sequence_length = metadata.get('sequence_length', 24)
        forecast_horizon = metadata.get('forecast_horizon', 24)
        target_column = metadata.get('target_column', 'OT')
        
        # Check if model type is specified in metadata
        model_type = metadata.get('model_type', None)
        if model_type:
            print(f"Model type from metadata: {model_type}")
    else:
        print("No metadata found in history, using default values")
        sequence_length = 24
        forecast_horizon = 24
        target_column = 'OT'
    
    feature_columns = history.get('feature_names', None)
    
    # Load test data
    print(f"Loading test data from {test_features_path}")
    test_df = pd.read_csv(test_features_path, index_col=0, parse_dates=True)
    
    # If feature_columns not in history, use all columns except target
    if feature_columns is None:
        feature_columns = [col for col in test_df.columns if col != target_column]
    
    # Print feature matching information
    available_columns = test_df.columns.tolist()
    missing_columns = [col for col in feature_columns if col not in available_columns]
    present_columns = [col for col in feature_columns if col in available_columns]
    
    print(f"Model requires {len(feature_columns)} features, found {len(present_columns)} in dataset")
    print(f"Missing {len(missing_columns)} features that will be created or filled with zeros")
    
    # Make predictions with enhanced feature handling
    predictions_df = predict_dataset(
        model=model,
        df=test_df,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
        target_column=target_column,
        scaler=scaler,
        freq='h'
    )
    
    # Save predictions
    print(f"Saving predictions to {predictions_path}")
    predictions_df.to_csv(predictions_path)
    
    print("Prediction completed")