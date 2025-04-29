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

def prepare_data_for_prediction(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str],
    scaler: Optional[Any] = None
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
    
    # Scale data if scaler is provided
    if scaler:
        logger.info("Scaling data")
        scaled_values = scaler.transform(data)
        data = pd.DataFrame(scaled_values, index=data.index, columns=data.columns)
    
    # Extract feature columns
    if not all(col in data.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in data.columns]
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    X_values = data[feature_columns].values
    
    # Create sequences
    X_sequences = []
    timestamps = []
    
    # For prediction, we create sequences but no targets (since we're predicting them)
    for i in range(len(data) - sequence_length + 1):
        X_sequences.append(X_values[i:i+sequence_length])
        # Store the timestamp of the last point in the input sequence
        timestamps.append(data.index[i+sequence_length-1])
    
    logger.info(f"Created {len(X_sequences)} prediction sequences")
    
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
    
    # Make predictions
    predictions = model.predict(X)
    
    # Denormalize predictions if scaler is provided
    if scaler and target_column_idx is not None:
        logger.info("Denormalizing predictions")
        
        # Create a dummy array with correct shape for inverse_transform
        # We'll only replace the target column with our predictions
        dummy = np.zeros((len(predictions), scaler.n_features_in_))
        
        # For each prediction (which can be multi-step)
        denormalized_predictions = np.zeros_like(predictions)
        
        for i in range(predictions.shape[1]):  # For each step in the forecast horizon
            # Replace the target column with the predictions for this step
            dummy_step = dummy.copy()
            dummy_step[:, target_column_idx] = predictions[:, i]
            
            # Inverse transform
            denorm_step = scaler.inverse_transform(dummy_step)
            
            # Extract just the target column
            denormalized_predictions[:, i] = denorm_step[:, target_column_idx]
        
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
    
    for i in range(forecast_horizon):
        # Predict the next step
        next_pred = model.predict(sequence)
        
        # Store the prediction for this step
        predictions[0, i] = next_pred[0, 0]  # Assuming single-step prediction
        
        # Update sequence for next iteration by shifting and adding new prediction
        # Remove the oldest time step
        sequence = sequence[:, 1:, :]
        
        # If we need to update more than just the target column, we'd need additional logic here
        # This simplified version only updates the target column
        new_step = sequence[:, -1, :].copy()
        new_step[0, target_column_idx] = next_pred[0, 0]
        
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
    freq: str = 'H'
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
        Frequency of timestamps, by default 'H' (hourly)
        
    Returns
    -------
    pd.DatetimeIndex
        Future timestamps
    """
    return pd.date_range(
        start=start_timestamp + pd.Timedelta(freq),
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
    freq: str = 'H',
    return_actual: bool = True
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
        Time series frequency, by default 'H' (hourly)
    return_actual : bool, optional
        Whether to include actual values in output, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and optionally actual values
    """
    logger.info(f"Predicting for dataset with {len(df)} rows")
    
    # Prepare data for prediction
    X, timestamps = prepare_data_for_prediction(
        df, sequence_length, feature_columns, scaler
    )
    
    # Get target column index for denormalization
    if scaler:
        target_idx = list(df.columns).index(target_column)
    else:
        target_idx = None
    
    # Make predictions
    predictions = make_predictions(model, X, scaler, target_idx)
    
    # Create result DataFrame
    result_dfs = []
    
    # For each prediction (which is multi-step)
    for i, timestamp in enumerate(timestamps):
        # Generate forecast timestamps
        forecast_times = generate_future_timestamps(
            timestamp, forecast_horizon, freq
        )
        
        # Create DataFrame for this forecast
        forecast_df = pd.DataFrame(
            predictions[i].reshape(forecast_horizon, 1),
            index=forecast_times,
            columns=['predicted']
        )
        
        # Add actual values if requested and available
        if return_actual:
            # Check if we have actual data for the forecast period
            actual_data = df[target_column].loc[forecast_times] if forecast_times[0] in df.index else None
            
            if actual_data is not None and len(actual_data) == forecast_horizon:
                forecast_df['actual'] = actual_data.values
        
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
    model_path = model_dir / "lstm_model.keras"
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
    
    # Get metadata from history
    if history and 'metadata' in history:
        metadata = history['metadata']
        sequence_length = metadata.get('sequence_length', 24)
        forecast_horizon = metadata.get('forecast_horizon', 24)
        target_column = metadata.get('target_column', 'OT')
    else:
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
    
    # Make predictions
    predictions_df = predict_dataset(
        model=model,
        df=test_df,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
        target_column=target_column,
        scaler=scaler,
        freq='H'  # Hourly data
    )
    
    # Save predictions
    print(f"Saving predictions to {predictions_path}")
    predictions_df.to_csv(predictions_path)
    
    print("Prediction completed")