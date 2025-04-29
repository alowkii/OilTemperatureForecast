"""
Functions for creating and training LSTM models for oil temperature forecasting.

This module contains functions to:
1. Create data sequences for time series forecasting
2. Build LSTM model architectures
3. Train models with appropriate callbacks
4. Save trained models and training history
"""

import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for reproducibility

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_sequences(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    forecast_horizon: int,
    step: int = 1,
    feature_columns: Optional[List[str]] = None,
    handle_nans: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create input/output sequences for time series forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target
    target_column : str
        Name of the target column to forecast
    sequence_length : int
        Number of past time steps to use as input sequence
    forecast_horizon : int
        Number of future time steps to predict
    step : int, optional
        Step size between consecutive sequences, by default 1
    feature_columns : List[str], optional
        List of feature columns to include, by default None (uses all columns except target)
    handle_nans : bool, optional
        Whether to handle NaN values before creating sequences, by default True
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        X sequences, y sequences, and list of feature names
    """
    logger.info(f"Creating sequences with length {sequence_length}, horizon {forecast_horizon}, step {step}")
    
    if feature_columns is None:
        # Use all columns except target as features
        feature_columns = [col for col in df.columns if col != target_column]
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Ensure all feature columns exist in DataFrame
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_columns}")
    
    # Create copies to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by index
    data = data.sort_index()
    
    # Check for NaN values in feature columns and target
    feature_nan_count = data[feature_columns].isna().sum().sum()
    target_nan_count = data[target_column].isna().sum()
    
    if feature_nan_count > 0 or target_nan_count > 0:
        logger.warning(f"Found {feature_nan_count} NaN values in features and {target_nan_count} in target")
        
        if handle_nans:
            logger.info("Handling NaN values in features and target")
            
            # Handle NaNs in features
            if feature_nan_count > 0:
                # First try forward fill with a limit
                data[feature_columns] = data[feature_columns].ffill(limit=12)
                
                # Then use interpolation for remaining NaNs
                if data[feature_columns].isna().sum().sum() > 0:
                    data[feature_columns] = data[feature_columns].interpolate(method='time')
                
                # Finally use backward fill for any remaining NaNs (usually at beginning)
                if data[feature_columns].isna().sum().sum() > 0:
                    data[feature_columns] = data[feature_columns].bfill()
            
            # Handle NaNs in target - important for training sequences
            if target_nan_count > 0:
                # Apply same approach to target
                data[target_column] = data[target_column].ffill(limit=12)
                
                if data[target_column].isna().sum() > 0:
                    data[target_column] = data[target_column].interpolate(method='time')
                
                if data[target_column].isna().sum() > 0:
                    data[target_column] = data[target_column].bfill()
            
            # Check if we still have NaNs
            remaining_feature_nans = data[feature_columns].isna().sum().sum()
            remaining_target_nans = data[target_column].isna().sum()
            
            if remaining_feature_nans > 0 or remaining_target_nans > 0:
                logger.warning(f"Still have {remaining_feature_nans} NaN values in features and {remaining_target_nans} in target after handling")
    
    # Get target and feature values
    y_values = data[target_column].values
    X_values = data[feature_columns].values
    
    X_sequences = []
    y_sequences = []
    
    # Create sequences
    for i in range(0, len(data) - sequence_length - forecast_horizon + 1, step):
        X_seq = X_values[i:i+sequence_length]
        y_seq = y_values[i+sequence_length:i+sequence_length+forecast_horizon]
        
        # Skip sequences with NaN values
        if np.isnan(X_seq).any() or np.isnan(y_seq).any():
            logger.debug(f"Skipping sequence at position {i} due to NaN values")
            continue
            
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    logger.info(f"Created {len(X_sequences)} sequences")
    
    if len(X_sequences) == 0:
        logger.warning("No valid sequences could be created due to NaN values")
        # Return empty arrays with correct shapes
        return (np.empty((0, sequence_length, len(feature_columns))), 
                np.empty((0, forecast_horizon)), 
                feature_columns)
    
    return np.array(X_sequences), np.array(y_sequences), feature_columns

def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    shuffle: bool = False,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into training and validation sets.
    
    Parameters
    ----------
    X : np.ndarray
        Input sequences
    y : np.ndarray
        Output sequences
    val_size : float, optional
        Fraction of data to use for validation, by default 0.2
    shuffle : bool, optional
        Whether to shuffle data before splitting, by default False
        (For time series, shuffling is usually not recommended)
    random_state : int, optional
        Random state for reproducibility if shuffling, by default None
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_val, y_train, y_val
    """
    # For time series, we typically split sequentially
    if not shuffle:
        split_idx = int(len(X) * (1 - val_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Split data into train ({len(X_train)} sequences) and validation ({len(X_val)} sequences)")
        return X_train, X_val, y_train, y_val
    
    # If shuffling is requested (less common for time series)
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - val_size))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    logger.info(f"Split data with shuffling into train ({len(X_train)} sequences) and validation ({len(X_val)} sequences)")
    return X_train, X_val, y_train, y_val

def create_lstm_model(
    input_shape: Tuple[int, int],
    output_dim: int,
    lstm_units: Union[int, List[int]] = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    bidirectional: bool = False
) -> tf.keras.Model:
    """
    Create an LSTM model for time series forecasting.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input sequences (sequence_length, n_features)
    output_dim : int
        Dimension of output (forecast horizon)
    lstm_units : Union[int, List[int]], optional
        Number of units in LSTM layer(s), by default 64
        If a list, creates multiple LSTM layers with the specified units
    dropout_rate : float, optional
        Dropout rate for regularization, by default 0.2
    learning_rate : float, optional
        Learning rate for Adam optimizer, by default 0.001
    bidirectional : bool, optional
        Whether to use bidirectional LSTM layers, by default False
        
    Returns
    -------
    tf.keras.Model
        Compiled LSTM model
    """
    logger.info(f"Creating LSTM model with input shape {input_shape} and output dim {output_dim}")
    
    model = Sequential()
    
    # Convert single value to list for consistent processing
    if isinstance(lstm_units, int):
        lstm_units = [lstm_units]
    
    # Add LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1  # Return sequences for all but last LSTM layer
        
        if i == 0:
            # First layer needs input shape
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences), input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            # Subsequent layers
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
        
        # Add dropout after each LSTM layer
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(output_dim))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Model created: {model.summary()}")
    return model

def create_advanced_lstm_model(
    input_shape: Tuple[int, int],
    output_dim: int,
    lstm_units: List[int] = [128, 64],
    dense_units: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.1,
    learning_rate: float = 0.001,
    bidirectional: bool = True
) -> tf.keras.Model:
    """
    Create an advanced LSTM model with multiple layers and additional features.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input sequences (sequence_length, n_features)
    output_dim : int
        Dimension of output (forecast horizon)
    lstm_units : List[int], optional
        Units in each LSTM layer, by default [128, 64]
    dense_units : List[int], optional
        Units in dense layers after LSTM, by default [64, 32]
    dropout_rate : float, optional
        Dropout rate for dense layers, by default 0.2
    recurrent_dropout : float, optional
        Recurrent dropout rate for LSTM layers, by default 0.1
    learning_rate : float, optional
        Learning rate for Adam optimizer, by default 0.001
    bidirectional : bool, optional
        Whether to use bidirectional LSTM, by default True
        
    Returns
    -------
    tf.keras.Model
        Compiled advanced LSTM model
    """
    logger.info("Creating advanced LSTM model")
    
    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        
        if bidirectional:
            x = Bidirectional(
                LSTM(units, return_sequences=return_sequences, recurrent_dropout=recurrent_dropout)
            )(x)
        else:
            x = LSTM(units, return_sequences=return_sequences, recurrent_dropout=recurrent_dropout)(x)
        
        x = Dropout(dropout_rate)(x)
    
    # Dense layers
    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(output_dim)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Advanced model created: {model.summary()}")
    return model

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10,
    min_delta: float = 0.001,
    reduce_lr_factor: float = 0.5,
    reduce_lr_patience: int = 5,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a model with early stopping and learning rate reduction.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model to train
    X_train : np.ndarray
        Training input sequences
    y_train : np.ndarray
        Training output sequences
    X_val : np.ndarray
        Validation input sequences
    y_val : np.ndarray
        Validation output sequences
    batch_size : int, optional
        Batch size for training, by default 32
    epochs : int, optional
        Maximum number of epochs, by default 100
    patience : int, optional
        Patience for early stopping, by default 10
    min_delta : float, optional
        Minimum improvement for early stopping, by default 0.001
    reduce_lr_factor : float, optional
        Factor to reduce learning rate, by default 0.5
    reduce_lr_patience : int, optional
        Patience for learning rate reduction, by default 5
    model_path : str, optional
        Path to save best model during training, by default None
        
    Returns
    -------
    Dict[str, Any]
        Training history and additional information
    """
    logger.info(f"Training model with {len(X_train)} training samples and {len(X_val)} validation samples")
    
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        verbose=1,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint if path is provided
    if model_path:
        model_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Get information about training
    epochs_completed = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = min(history.history['val_mae'])
    
    logger.info(f"Training completed after {epochs_completed} epochs")
    logger.info(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}, val_mae: {best_val_mae:.4f}")
    
    # Return history and additional information
    return {
        'history': history.history,
        'epochs_completed': epochs_completed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae
    }

def save_model(
    model: tf.keras.Model,
    training_info: Dict[str, Any],
    model_path: str,
    scaler_path: Optional[str] = None,
    scaler: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model, training information, and associated metadata.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model to save
    training_info : Dict[str, Any]
        Training history and information
    model_path : str
        Path to save the model
    scaler_path : str, optional
        Path to save the scaler, by default None
    scaler : Any, optional
        Fitted scaler for preprocessing, by default None
    feature_names : List[str], optional
        Names of features used in the model, by default None
    metadata : Dict[str, Any], optional
        Additional metadata to save, by default None
        
    Returns
    -------
    None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create reports directory if needed
    history_plot_dir = os.path.join(os.path.dirname(model_path), "report")
    os.makedirs(history_plot_dir, exist_ok=True)
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    model.save(model_path)
    
    # Save training history and metrics
    history_path = os.path.splitext(model_path)[0] + '_history.json'
    
    # Fix for NumPy types not being JSON serializable
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        else:
            return obj
    
    # Extract history dictionary and convert NumPy types
    history_dict = {}
    for k, v in training_info['history'].items():
        history_dict[k] = [float(val) for val in v]
    
    # Create serializable training info
    serializable_info = {
        'history': history_dict,
        'epochs_completed': int(training_info['epochs_completed']),
        'best_epoch': int(training_info['best_epoch']),
        'best_val_loss': float(training_info['best_val_loss']),
        'best_val_mae': float(training_info['best_val_mae'])
    }
    
    # Add metadata if provided
    if metadata:
        serializable_info['metadata'] = convert_to_serializable(metadata)
    
    # Add feature names if provided
    if feature_names:
        serializable_info['feature_names'] = [str(f) for f in feature_names]
    
    # Save training info
    logger.info(f"Saving training history to {history_path}")
    with open(history_path, 'w') as f:
        json.dump(serializable_info, f, indent=4)
    
    # Save scaler if provided
    if scaler and scaler_path:
        logger.info(f"Saving scaler to {scaler_path}")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    # Plot and save training history
    history_plot_path = os.path.join(history_plot_dir, "training_history.png")
    plot_training_history(
        history=history_dict,
        output_path=history_plot_path
    )

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot training and validation loss/metrics.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Training history dictionary
    output_path : str, optional
        Path to save the plot, by default None (display only)
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
        
    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(2, 1, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Training history plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    project_dir = Path(__file__).resolve().parents[2]
    
    # Print the detected project directory
    print(f"Project directory: {project_dir}")
    
    # Define paths
    features_path = project_dir / "Data" / "features" / "train_features.csv"
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "lstm_model.keras"
    scaler_path = model_dir / "scaler.pkl"
    history_plot_path = model_dir / "report" / "training_history.png"
    
    # Load feature-engineered data
    print(f"Loading data from {features_path}")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    # Parameters
    target_column = 'OT'
    sequence_length = 24  # Use 24 hours of past data
    forecast_horizon = 24  # Predict next 24 hours
    
    # Exclude some features if needed
    excluded_features = []
    feature_columns = [col for col in df.columns if col != target_column and col not in excluded_features]
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    # Create sequences
    X, y, features_used = create_sequences(
        df_scaled,
        target_column=target_column,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns
    )
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = split_train_val(X, y, val_size=0.2)
    
    # Create model
    model = create_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        output_dim=forecast_horizon,
        lstm_units=[128, 64],
        dropout_rate=0.2,
        learning_rate=0.001,
        bidirectional=True
    )
    
    # Train model
    training_info = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=32,
        epochs=100,
        patience=10,
        model_path=str(model_path)
    )
    
    # Plot training history
    plot_training_history(
        history=training_info['history'],
        output_path=str(history_plot_path)
    )
    
    # Save model and associated artifacts
    save_model(
        model=model,
        training_info=training_info,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        scaler=scaler,
        feature_names=features_used,
        metadata={
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'target_column': target_column
        }
    )
    
    print("Model training and saving completed")