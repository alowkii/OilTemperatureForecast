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
from tensorflow.keras.layers import Dense, LSTM, Dropout, Attention, Bidirectional, Input, Concatenate, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.make_dataset import analyze_temperature_distribution

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess_with_enhanced_scaling(
    df: pd.DataFrame,
    target_column: str = 'OT',
    scaling_method: str = 'robust',
    include_extremes_analysis: bool = True,
    temp_stats_path: Optional[str] = None  # Add this parameter
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """
    Enhanced preprocessing with better scaling and extreme value handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features
    target_column : str, optional
        Target column name, by default 'OT'
    scaling_method : str, optional
        Scaling method to use ('robust', 'minmax', or 'standard'), by default 'robust'
    include_extremes_analysis : bool, optional
        Whether to perform analysis of extreme values, by default True
    temp_stats_path : str, optional
        Path to pre-computed temperature statistics, by default None
        
    Returns
    -------
    Tuple[pd.DataFrame, Any, Dict[str, Any]]
        Scaled DataFrame, scaler, and metadata with extremes analysis
    """
    logger.info(f"Performing enhanced preprocessing with {scaling_method} scaling")
    
    # Make a copy of input data
    df_copy = df.copy()
    
    # Analyze temperature distribution
    extremes_info = {}
    
    # Try to load pre-computed temperature statistics if path is provided
    loaded_temp_stats = None
    if temp_stats_path and os.path.exists(temp_stats_path):
        try:
            with open(temp_stats_path, 'r') as f:
                loaded_temp_stats = json.load(f)
            logger.info(f"Loaded temperature statistics from {temp_stats_path}")
        except Exception as e:
            logger.warning(f"Failed to load temperature statistics: {str(e)}")
    
    if include_extremes_analysis and target_column in df_copy.columns:
        # Use pre-computed statistics or calculate now
        if loaded_temp_stats:
            temp_stats = loaded_temp_stats
            logger.info("Using pre-computed temperature statistics")
        else:
            # Run the analysis function
            temp_stats = analyze_temperature_distribution(df_copy, target_column)
            logger.info("Computed temperature statistics")
        
        extremes_info['temperature_stats'] = temp_stats
        extremes_info['extreme_threshold'] = temp_stats['p95']
        extremes_info['extreme_count'] = int((df_copy[target_column] >= temp_stats['p95']).sum())
        extremes_info['extreme_percentage'] = float((df_copy[target_column] >= temp_stats['p95']).mean() * 100)
        
        logger.info(f"Dataset contains {extremes_info['extreme_count']} extreme temperature values "
                   f"(≥ {extremes_info['extreme_threshold']:.2f}°C), "
                   f"which is {extremes_info['extreme_percentage']:.2f}% of the data")
    
    # Choose and apply the appropriate scaler
    if scaling_method == 'robust':
        # RobustScaler is better for data with outliers
        scaler = RobustScaler()
        logger.info("Using RobustScaler which is more appropriate for data with outliers")
    elif scaling_method == 'minmax':
        # MinMaxScaler with wider range to better preserve differences in extreme values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        logger.info("Using MinMaxScaler with range (-1, 1) to better preserve extreme values")
    else:
        # Standard scaler
        scaler = StandardScaler()
        logger.info("Using StandardScaler")
    
    # Fit and transform the data
    scaled_data = pd.DataFrame(
        scaler.fit_transform(df_copy),
        index=df_copy.index,
        columns=df_copy.columns
    )
    
    logger.info(f"Scaling complete, data shape: {scaled_data.shape}")
    
    # Create metadata dictionary
    metadata = {
        'scaling_method': scaling_method,
        'extremes_info': extremes_info,
        'feature_names': list(df_copy.columns),
        'scaler_type': type(scaler).__name__
    }
    
    return scaled_data, scaler, metadata

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by handling infinity, NaN values, and extreme outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    logger.info("Cleaning dataset to remove infinities and extreme values")
    
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Count initial issues
    initial_inf_count = np.isinf(cleaned_df.values).sum()
    initial_nan_count = np.isnan(cleaned_df.values).sum()
    
    logger.info(f"Initial data state: {initial_inf_count} infinities, {initial_nan_count} NaNs")
    
    # Replace infinity values with NaN
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaNs with column median and cap extreme values
    for column in cleaned_df.columns:
        # Skip columns that are all NaN
        if cleaned_df[column].isna().all():
            logger.warning(f"Column '{column}' contains all NaNs and will be dropped")
            continue
            
        # Get column statistics (ignoring NaNs)
        col_median = cleaned_df[column].median()
        col_std = cleaned_df[column].std()
        
        # If std is NaN or 0, use a small default value
        if np.isnan(col_std) or col_std == 0:
            col_std = 1e-5
            
        # Define acceptable range (10 std from median)
        lower_bound = col_median - 10 * col_std
        upper_bound = col_median + 10 * col_std
        
        # Replace NaNs with median - FIXED: assigning back to DataFrame
        cleaned_df[column] = cleaned_df[column].fillna(col_median)
        
        # Cap extreme values - FIXED: more direct approach
        cleaned_df.loc[cleaned_df[column] < lower_bound, column] = lower_bound
        cleaned_df.loc[cleaned_df[column] > upper_bound, column] = upper_bound
    
    # Verify no infinities or NaNs remain
    final_inf_count = np.isinf(cleaned_df.values).sum()
    final_nan_count = np.isnan(cleaned_df.values).sum()
    
    if final_inf_count > 0 or final_nan_count > 0:
        logger.warning(f"After cleaning: {final_inf_count} infinities, {final_nan_count} NaNs remain")
        
        # Last resort - drop problematic columns
        for column in cleaned_df.columns:
            if np.isinf(cleaned_df[column]).any() or np.isnan(cleaned_df[column]).any():
                logger.warning(f"Dropping problematic column: '{column}'")
                cleaned_df = cleaned_df.drop(columns=[column])
    
    # Final verification
    assert not np.isinf(cleaned_df.values).any(), "Infinity values still present after cleaning"
    assert not np.isnan(cleaned_df.values).any(), "NaN values still present after cleaning"
    
    logger.info(f"Cleaning complete: {cleaned_df.shape[1]} columns remaining")
    return cleaned_df

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

# Add the custom loss function definition
def temporal_weighted_mse():
    """
    Custom loss function that places higher importance on certain time steps.
    """
    def loss_fn(y_true, y_pred):
        # Calculate squared error
        squared_error = tf.square(y_true - y_pred)
        
        # Set weights for different time steps in the prediction horizon
        # Higher weights for early predictions (first 6 hours) and day boundaries
        weights = tf.ones_like(y_true)
        
        # Emphasize first 6 steps (critical short-term forecast)
        weights = tf.tensor_scatter_nd_update(
            weights,
            indices=tf.constant([[i, j] for i in range(tf.shape(y_true)[0]) for j in range(6)]),
            updates=tf.constant([1.5] * (tf.shape(y_true)[0] * 6))
        )
        
        # Emphasize day boundaries (hours 23-24)
        if tf.shape(y_true)[1] >= 24:
            weights = tf.tensor_scatter_nd_update(
                weights,
                indices=tf.constant([[i, j] for i in range(tf.shape(y_true)[0]) for j in range(22, 24)]),
                updates=tf.constant([1.3] * (tf.shape(y_true)[0] * 2))
            )
        
        # Apply weights and calculate mean
        weighted_squared_error = squared_error * weights
        return tf.reduce_mean(weighted_squared_error)
    
    return loss_fn

def create_encoder_decoder_model(
    input_shape: Tuple[int, int],
    output_dim: int,
    encoder_units: List[int] = [128, 128],
    decoder_units: List[int] = [128, 64],
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.2,
    use_attention: bool = True
) -> Union[tf.keras.Model, Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]]:
    """
    Create an encoder-decoder architecture with optional attention mechanism.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input sequences (sequence_length, n_features)
    output_dim : int
        Number of steps to forecast
    encoder_units : List[int], optional
        Units in each encoder LSTM layer, by default [128, 128]
    decoder_units : List[int], optional
        Units in each decoder LSTM layer, by default [128, 64]
    dropout_rate : float, optional
        Dropout rate after layers, by default 0.3
    recurrent_dropout : float, optional
        Recurrent dropout within LSTM cells, by default 0.2
    use_attention : bool, optional
        Whether to use attention mechanism, by default True
        
    Returns
    -------
    Union[tf.keras.Model, Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]]
        If use_attention is True: Tuple of (training model, encoder model, decoder model)
        Otherwise: Compiled encoder-decoder model
    """
    logger.info(f"Creating encoder-decoder model with{'out' if not use_attention else ''} attention")
    
    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_inputs')
    encoder = encoder_inputs
    encoder_states = []
    
    for i, units in enumerate(encoder_units):
        return_sequences = i < len(encoder_units) - 1 or use_attention
        
        lstm_layer = Bidirectional(
            LSTM(
                units,
                return_sequences=return_sequences,
                return_state=True,
                recurrent_dropout=recurrent_dropout,
                name=f'encoder_lstm_{i}'
            )
        )
        
        if i == 0:
            encoder, forward_h, forward_c, backward_h, backward_c = lstm_layer(encoder)
        else:
            encoder, forward_h, forward_c, backward_h, backward_c = lstm_layer(encoder)
        
        encoder = BatchNormalization(name=f'encoder_bn_{i}')(encoder)

        # Concatenate forward and backward states
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
    
    # Attention mechanism
    if use_attention:
        # Create decoder input (will be used later for teacher forcing)
        decoder_inputs = Input(shape=(None, 1), name='decoder_inputs')
        
        # Initialize decoder with encoder states
        decoder = decoder_inputs
        
        # Fix: Use decoder_units[0] * 2 to match the bidirectional encoder output size
        decoder_lstm = LSTM(
            decoder_units[0] * 2,  # Multiply by 2 to match bidirectional encoder state size
            return_sequences=True,
            return_state=True,
            recurrent_dropout=recurrent_dropout,
            name='decoder_lstm'
        )
        
        # Initial decoder state is the encoder final state
        decoder_outputs, _, _ = decoder_lstm(decoder, initial_state=encoder_states)
        decoder_outputs = BatchNormalization(name='decoder_bn')(decoder_outputs)

        # Apply attention
        attention_layer = Attention(name='attention_layer')
        context_vector = attention_layer([decoder_outputs, encoder])
        
        # Concatenate context vector and decoder output
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        
        # Apply dropout
        decoder_combined = Dropout(dropout_rate)(decoder_combined)
        
        # Add dense layers
        for i, units in enumerate(decoder_units[1:]):
            decoder_combined = Dense(
                units,
                activation='relu',
                name=f'decoder_dense_{i}'
            )(decoder_combined)
            decoder_combined = Dropout(dropout_rate)(decoder_combined)
        
        # Output layer
        outputs = Dense(output_dim, name='output_layer')(decoder_combined)
        
        # Define training model (with teacher forcing)
        train_model = Model([encoder_inputs, decoder_inputs], outputs)
        
        # Define inference encoder model
        encoder_model = Model(encoder_inputs, [encoder] + encoder_states)
        
        # Define inference decoder model - Fix: Use decoder_units[0] * 2 here too
        decoder_state_input_h = Input(shape=(decoder_units[0] * 2,))  # Match bidirectional size
        decoder_state_input_c = Input(shape=(decoder_units[0] * 2,))  # Match bidirectional size
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        
        encoder_outputs_input = Input(shape=(input_shape[0], encoder_units[-1] * 2))  # *2 for bidirectional
        context_vector = attention_layer([decoder_outputs, encoder_outputs_input])
        
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_combined = Dropout(dropout_rate)(decoder_combined)
        
        for i, units in enumerate(decoder_units[1:]):
            decoder_combined = Dense(
                units,
                activation='relu',
                name=f'decoder_dense_{i}_inference'
            )(decoder_combined)
            decoder_combined = Dropout(dropout_rate)(decoder_combined)
        
        decoder_outputs = Dense(1, name='output_layer_inference')(decoder_combined)
        
        decoder_model = Model(
            [decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
        # Compile model
        train_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return train_model, encoder_model, decoder_model
    
    else:
        # Simpler model without attention mechanism
        decoder = encoder if len(encoder_units) == len(decoder_units) else None
        
        for i, units in enumerate(decoder_units):
            return_sequences = True  # We want sequences for all decoder layers
            
            if decoder is None:
                # First decoder layer, initialize with encoder states
                decoder_lstm = LSTM(
                    units,
                    return_sequences=return_sequences,
                    recurrent_dropout=recurrent_dropout,
                    name=f'decoder_lstm_{i}'
                )
                decoder = decoder_lstm(encoder_states[0], initial_state=encoder_states)
            else:
                decoder_lstm = LSTM(
                    units,
                    return_sequences=return_sequences,
                    recurrent_dropout=recurrent_dropout,
                    name=f'decoder_lstm_{i}'
                )
                decoder = decoder_lstm(decoder)
            
            decoder = Dropout(dropout_rate)(decoder)
        
        # Flatten for output layer
        decoder = Flatten()(decoder)
        
        # Add dense layers to map to output dimension
        decoder = Dense(128, activation='relu')(decoder)
        decoder = Dropout(dropout_rate)(decoder)
        
        outputs = Dense(output_dim)(decoder)
        
        model = Model(encoder_inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
class ExtremeValueLoss(tf.keras.losses.Loss):
    """
    Custom loss function that gives higher weight to errors on extreme values.
    
    Parameters
    ----------
    threshold : float
        Temperature threshold above which to apply higher weight
    high_temp_weight : float
        Weight multiplier for temperatures above threshold
    reduction : str
        Reduction method ('auto', 'none', 'sum', 'mean')
    name : str
        Name of the loss function
    """
    
    def __init__(
        self,
        threshold: float = 40.0,
        high_temp_weight: float = 2.0,
        reduction: str = 'auto',
        name: str = 'extreme_value_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.threshold = threshold
        self.high_temp_weight = high_temp_weight
    
    def call(self, y_true, y_pred):
        # Calculate MSE
        mse = tf.square(y_true - y_pred)
        
        # Create mask for extreme values
        extreme_mask = tf.cast(y_true > self.threshold, tf.float32)
        
        # Apply higher weight to errors on extreme values
        weighted_mse = mse * (1 + extreme_mask * (self.high_temp_weight - 1))
        
        return weighted_mse
    
def train_with_horizon_weighting(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 15,
    min_delta: float = 0.0001,
    horizon_weights: Optional[np.ndarray] = None,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train model with custom horizon weighting to prioritize earlier predictions.
    
    Parameters
    ----------
    model : tf.keras.Model or Tuple
        Model to train, or tuple of (train_model, encoder_model, decoder_model)
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    batch_size, epochs, patience, min_delta : standard training parameters
    horizon_weights : np.ndarray, optional
        Weights for each forecast horizon (higher for more important horizons)
    model_path : str, optional
        Path to save best model
        
    Returns
    -------
    Dict[str, Any]
        Training history and information
    """
    logger.info(f"Training model with horizon weighting")
    
    # Handle case where model is a tuple (train_model, encoder_model, decoder_model)
    is_encoder_decoder = isinstance(model, tuple)
    if is_encoder_decoder:
        train_model = model[0]  # Extract just the training model
    else:
        train_model = model
    
    # Create horizon weights if not provided
    if horizon_weights is None:
        # Give higher weights to earlier horizons (exponential decay)
        forecast_horizon = y_train.shape[1]
        horizon_weights = np.exp(-0.1 * np.arange(forecast_horizon))
        horizon_weights = horizon_weights / np.sum(horizon_weights)
        logger.info(f"Using exponential decay for horizon weights")
    
    # Create callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        verbose=1,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint
    if model_path:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
    
    # For encoder-decoder models with attention, we need to prepare decoder inputs
    if is_encoder_decoder:
        # For teacher forcing, we need to provide the target sequence as decoder input
        # but shifted by one timestep (starts with zeros and ends without the last value)
        
        # Create decoder inputs (shifted by one time step)
        # For training data
        decoder_input_train = np.zeros((len(y_train), 1, 1))  # Initial zero timestep
        
        # For validation data
        decoder_input_val = np.zeros((len(y_val), 1, 1))  # Initial zero timestep
        
        # Train the model with the appropriate inputs
        history = train_model.fit(
            [X_train, decoder_input_train],  # Model expects [encoder_input, decoder_input]
            y_train,
            validation_data=([X_val, decoder_input_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # For standard models, use the custom loss wrapper approach
        # Store original loss
        original_loss = train_model.loss
        
        # Create horizon-weighted wrapper
        class HorizonWeightedLoss(tf.keras.losses.Loss):
            def __init__(self, base_loss, horizon_weights):
                super().__init__(name='horizon_weighted_loss')
                self.base_loss = base_loss
                self.horizon_weights = tf.constant(horizon_weights, dtype=tf.float32)
            
            def call(self, y_true, y_pred):
                # Calculate base loss for each sample and horizon
                # Shape: (batch_size, horizon)
                per_horizon_loss = tf.square(y_true - y_pred)
                
                # Apply horizon weights
                weighted_loss = per_horizon_loss * self.horizon_weights
                
                # Sum across horizons, mean across batch
                return tf.reduce_mean(tf.reduce_sum(weighted_loss, axis=1))
        
        # Create horizon-weighted wrapper
        weighted_loss = HorizonWeightedLoss(original_loss, horizon_weights)
        
        # Set model loss to the custom loss
        train_model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss=temporal_weighted_mse(),
            metrics=['mae']
        )
        
        # Train the model
        history = train_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Restore original loss
        train_model.compile(
            optimizer=train_model.optimizer,
            loss=original_loss,
            metrics=train_model.metrics
        )
    
    # Get training information
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
        'best_val_mae': best_val_mae,
        'horizon_weights': horizon_weights.tolist()
    }

def create_balanced_sequences(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    forecast_horizon: int,
    feature_columns: Optional[List[str]] = None,
    extreme_threshold: Optional[float] = None,
    extreme_sample_weight: float = 2.0,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create sequences with balanced representation of extreme values.
    
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
    feature_columns : List[str], optional
        List of feature columns to include, by default None (uses all columns except target)
    extreme_threshold : float, optional
        Threshold for defining extreme values, by default None (uses 95th percentile)
    extreme_sample_weight : float, optional
        Weight for extreme samples, by default 2.0
    step : int, optional
        Step size between consecutive sequences, by default 1
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]
        X sequences, y sequences, sample weights, and list of feature names
    """
    logger.info(f"Creating balanced sequences with length {sequence_length}, horizon {forecast_horizon}")
    
    # First create regular sequences
    X, y, features = create_sequences(
        df, target_column, sequence_length, forecast_horizon, 
        step=step, feature_columns=feature_columns
    )
    
    # If no extreme threshold provided, use 95th percentile
    if extreme_threshold is None:
        if target_column in df.columns:
            extreme_threshold = df[target_column].quantile(0.95)
            logger.info(f"Using 95th percentile ({extreme_threshold:.2f}) as extreme threshold")
        else:
            logger.warning(f"Target column '{target_column}' not found, cannot determine extreme threshold")
            extreme_threshold = float('inf')  # Default that won't match anything
    
    # Calculate weights for each sequence
    weights = np.ones(len(y))
    
    # Check for extreme values in target sequences
    for i in range(len(y)):
        # If any value in the target sequence is extreme, give it higher weight
        if np.any(y[i] >= extreme_threshold):
            weights[i] = extreme_sample_weight
    
    # Count extreme sequences
    extreme_count = np.sum(weights > 1.0)
    logger.info(f"Created {len(X)} sequences, including {extreme_count} with extreme values "
               f"({extreme_count/len(X)*100:.2f}%)")
    
    return X, y, weights, features

def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    shuffle: bool = False,  # Set default to False
    random_state: Optional[int] = None,
    timestamps: Optional[np.ndarray] = None  # Add timestamps parameter
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into training and validation sets.
    Prioritizes time-based splitting for time series data.
    
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
    random_state : int, optional
        Random state for reproducibility if shuffling, by default None
    timestamps : np.ndarray, optional
        Timestamps for sequences, by default None
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_val, y_train, y_val
    """
    # For time series, we typically split sequentially
    if not shuffle:
        if timestamps is not None:
            # If timestamps are provided, sort by time first
            logger.info("Performing time-based split with timestamps")
            sorted_indices = np.argsort(timestamps)
            X = X[sorted_indices]
            y = y[sorted_indices]
        
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
    features_path = project_dir / "data" / "features" / "train_features_enhanced.csv"
    temp_stats_path = project_dir / "data" / "preprocessed" / "temperature_stats.json"
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "lstm_encoder_decoder_model.keras"
    scaler_path = model_dir / "robust_scaler.pkl"
    history_plot_path = model_dir / "report" / "training_history.png"
    
    # Load feature-engineered data
    print(f"Loading data from {features_path}")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    # Parameters
    target_column = 'OT'
    sequence_length = 24  # Use 24 hours of past data
    forecast_horizon = 24  # Predict next 24 hours
    
    # Get extreme temperature threshold
    extreme_threshold = None
    if os.path.exists(temp_stats_path):
        try:
            with open(temp_stats_path, 'r') as f:
                temp_stats = json.load(f)
                extreme_threshold = temp_stats.get('p95', 45.0)
                print(f"Using extreme temperature threshold: {extreme_threshold}°C")
        except:
            extreme_threshold = 45.0
    else:
        extreme_threshold = df[target_column].quantile(0.95)
        print(f"Calculated extreme temperature threshold: {extreme_threshold}°C")
    
    # Clean the dataset to remove infinities and extreme values
    df = clean_dataset(df)
    
    # Preprocessing with enhanced scaling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    # Scale all columns 
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index, 
        columns=df.columns
    )
    
    # Exclude some features with low importance or high correlation
    # Note: Adjust this list based on feature importance analysis
    excluded_features = []
    feature_columns = [col for col in df.columns if col != target_column and col not in excluded_features]
    
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

    train_extreme_count = np.sum(np.any(y_train >= extreme_threshold, axis=1))
    val_extreme_count = np.sum(np.any(y_val >= extreme_threshold, axis=1))
    train_extreme_pct = (train_extreme_count / len(y_train)) * 100
    val_extreme_pct = (val_extreme_count / len(y_val)) * 100
    print(f"Extreme temps in training: {train_extreme_pct:.2f}% ({train_extreme_count} sequences)")
    print(f"Extreme temps in validation: {val_extreme_pct:.2f}% ({val_extreme_count} sequences)")
    if abs(train_extreme_pct - val_extreme_pct) > 5.0:
        print(f"WARNING: Large difference in extreme temperature distribution between train and validation sets!")
    
    # Create encoder-decoder model
    model_result = create_encoder_decoder_model(
        input_shape=(X.shape[1], X.shape[2]),
        output_dim=forecast_horizon,
        encoder_units=[196, 128],
        decoder_units=[128, 96, 64, 32],
        dropout_rate=0.25, # disabled dropout for simplicity
        recurrent_dropout=0.15,
        use_attention=True
    )
    
    # Create horizon weights (higher weights for earlier horizons and extreme temps)
    horizon_weights = np.exp(-0.07 * np.arange(forecast_horizon))
    horizon_weights = horizon_weights / np.sum(horizon_weights)
    
    # Train with horizon weighting to prioritize earlier predictions
    training_info = train_with_horizon_weighting(
        model=model_result,  # Pass the entire result (function will handle tuple)
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=64,
        epochs=100,
        patience=25,
        min_delta=0.0001,
        horizon_weights=horizon_weights,
        model_path=str(model_path)
    )
    
    # Extract the training model if model_result is a tuple
    if isinstance(model_result, tuple):
        train_model = model_result[0]
    else:
        train_model = model_result
    
    # Save model and artifacts
    save_model(
        model=train_model,  # Use the training model
        training_info=training_info,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        scaler=scaler,
        feature_names=features_used,
        metadata={
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'target_column': target_column,
            'extreme_threshold': extreme_threshold,
            'model_type': 'encoder_decoder_with_attention'
        }
    )