"""
Functions for visualizing transformer oil temperature data and model predictions.

This module contains functions to:
1. Create time series plots of raw and processed data
2. Visualize feature importance and relationships
3. Plot model predictions against actual values
4. Create heatmaps for temperature patterns
5. Generate interactive dashboards for exploring results
6. Analyze prediction errors by time, horizon, and temperature value
7. Compare different model performances
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def plot_time_series(
    df: pd.DataFrame,
    columns: List[str],
    title: str = 'Time Series Plot',
    figsize: Tuple[int, int] = (15, 8),
    output_path: Optional[str] = None,
    resample: Optional[str] = None,
    highlight_extremes: bool = False,
    extreme_threshold: Optional[float] = None,
    alpha: float = 0.8
) -> None:
    """
    Plot time series data for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    columns : List[str]
        List of columns to plot
    title : str, optional
        Plot title, by default 'Time Series Plot'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 8)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    resample : str, optional
        Resample frequency (e.g., 'D' for daily), by default None
    highlight_extremes : bool, optional
        Whether to highlight extreme values, by default False
    extreme_threshold : float, optional
        Threshold for extreme values, by default None (uses 95th percentile)
    alpha : float, optional
        Transparency for plot lines, by default 0.8
        
    Returns
    -------
    None
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info(f"Creating time series plot for {len(columns)} columns")
    
    # Create a copy of the data to avoid modifying the original
    data = df.copy()
    
    # Resample if specified
    if resample:
        data = data.resample(resample).mean()
        logger.info(f"Resampled data to {resample} frequency")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot each column
    for col in columns:
        if col not in data.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        plt.plot(data.index, data[col], label=col, alpha=alpha)
    
    # Highlight extreme values if requested
    if highlight_extremes and len(columns) == 1:
        col = columns[0]
        
        # Calculate threshold if not provided
        if extreme_threshold is None:
            extreme_threshold = data[col].quantile(0.95)
            logger.info(f"Using 95th percentile ({extreme_threshold:.2f}) as extreme threshold")
        
        # Create a mask for extreme values
        extremes = data[data[col] >= extreme_threshold]
        
        if not extremes.empty:
            plt.scatter(
                extremes.index, extremes[col],
                color='red', s=50, label=f'Extreme Values (>= {extreme_threshold:.2f})'
            )
            logger.info(f"Highlighted {len(extremes)} extreme values")
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Time series plot saved to {output_path}")
    else:
        plt.show()

def plot_prediction_vs_actual(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    title: str = 'Prediction vs Actual',
    figsize: Tuple[int, int] = (15, 8),
    output_path: Optional[str] = None,
    highlight_errors: bool = True,
    error_threshold: float = 5.0
) -> None:
    """
    Plot model predictions against actual values.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values, datetime index
    actual_column : str, optional
        Name of actual values column, by default 'actual'
    pred_column : str, optional
        Name of predicted values column, by default 'predicted'
    title : str, optional
        Plot title, by default 'Prediction vs Actual'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 8)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    highlight_errors : bool, optional
        Whether to highlight large errors, by default True
    error_threshold : float, optional
        Threshold for large errors, by default 5.0
        
    Returns
    -------
    None
    """
    if not isinstance(predictions_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if actual_column not in predictions_df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    logger.info(f"Creating prediction vs actual plot")
    
    # Make a copy of the data to avoid modifying the original
    df = predictions_df.copy()
    
    # Calculate absolute error
    df['abs_error'] = (df[actual_column] - df[pred_column]).abs()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot actual and predicted values
    plt.plot(df.index, df[actual_column], 'b-', label='Actual', alpha=0.7)
    plt.plot(df.index, df[pred_column], 'r-', label='Predicted', alpha=0.7)
    
    # Highlight large errors if requested
    if highlight_errors:
        large_errors = df[df['abs_error'] >= error_threshold]
        
        if not large_errors.empty:
            plt.scatter(
                large_errors.index, large_errors[actual_column],
                color='blue', s=50, alpha=0.7, label=f'Large Error Points (>= {error_threshold})'
            )
            plt.scatter(
                large_errors.index, large_errors[pred_column],
                color='red', s=50, alpha=0.7
            )
            
            # Connect large error points with lines
            for idx in large_errors.index:
                plt.plot(
                    [idx, idx],
                    [large_errors.loc[idx, actual_column], large_errors.loc[idx, pred_column]],
                    'k--', alpha=0.5
                )
                
            logger.info(f"Highlighted {len(large_errors)} points with large errors")
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Prediction vs actual plot saved to {output_path}")
    else:
        plt.show()

def plot_prediction_sequences(
    predictions_df: pd.DataFrame,
    num_sequences: int = 5,
    sequence_id_column: str = 'sequence_id',
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    random_seed: Optional[int] = None
) -> None:
    """
    Plot multiple prediction sequences to visualize model performance over time.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    num_sequences : int, optional
        Number of sequences to plot, by default 5
    sequence_id_column : str, optional
        Name of sequence ID column, by default 'sequence_id'
    actual_column : str, optional
        Name of actual values column, by default 'actual'
    pred_column : str, optional
        Name of predicted values column, by default 'predicted'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    random_seed : int, optional
        Random seed for sequence selection, by default None
        
    Returns
    -------
    None
    """
    if sequence_id_column not in predictions_df.columns:
        raise ValueError(f"Sequence ID column '{sequence_id_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    # Check if actual column exists (it might not for some predictions)
    has_actual = actual_column in predictions_df.columns
    
    logger.info(f"Creating plot for {num_sequences} prediction sequences")
    
    # Get unique sequence IDs
    sequence_ids = predictions_df[sequence_id_column].unique()
    
    # Limit number of sequences to plot
    num_sequences = min(num_sequences, len(sequence_ids))
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Randomly select sequence IDs to plot
    selected_ids = np.random.choice(sequence_ids, size=num_sequences, replace=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_sequences, 1, figsize=figsize, sharex=True)
    
    # Handle case with only one sequence
    if num_sequences == 1:
        axes = [axes]
    
    # Plot each selected sequence
    for i, seq_id in enumerate(selected_ids):
        seq_data = predictions_df[predictions_df[sequence_id_column] == seq_id].copy()
        
        # Sort by index
        seq_data = seq_data.sort_index()
        
        ax = axes[i]
        
        # Plot predicted values
        ax.plot(seq_data.index, seq_data[pred_column], 'r-', label='Predicted', alpha=0.7)
        
        # Plot actual values if available
        if has_actual:
            # Handle NaN values
            actual_mask = ~seq_data[actual_column].isna()
            if actual_mask.any():
                ax.plot(seq_data.index[actual_mask], seq_data.loc[actual_mask, actual_column], 
                       'b-', label='Actual', alpha=0.7)
        
        # Calculate metrics if actual values are available
        if has_actual and actual_mask.any():
            mae = np.mean(np.abs(seq_data.loc[actual_mask, actual_column] - seq_data.loc[actual_mask, pred_column]))
            rmse = np.sqrt(np.mean(np.square(seq_data.loc[actual_mask, actual_column] - seq_data.loc[actual_mask, pred_column])))
            title = f"Sequence {seq_id} - MAE: {mae:.2f}, RMSE: {rmse:.2f}"
        else:
            title = f"Sequence {seq_id}"
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Set common labels
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Temperature', va='center', rotation='vertical')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Prediction sequences plot saved to {output_path}")
    else:
        plt.show()

def plot_error_distribution(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    figsize: Tuple[int, int] = (15, 6),
    output_path: Optional[str] = None,
    n_bins: int = 50
) -> None:
    """
    Plot distribution of prediction errors.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    actual_column : str, optional
        Name of actual values column, by default 'actual'
    pred_column : str, optional
        Name of predicted values column, by default 'predicted'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 6)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    n_bins : int, optional
        Number of bins for histogram, by default 50
        
    Returns
    -------
    None
    """
    if actual_column not in predictions_df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    logger.info("Creating error distribution plot")
    
    # Make a copy of the data to avoid modifying the original
    df = predictions_df.copy()
    
    # Calculate errors
    df['error'] = df[actual_column] - df[pred_column]
    
    # Drop rows with NaN errors
    df = df.dropna(subset=['error'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot error histogram
    sns.histplot(df['error'], kde=True, bins=n_bins, ax=ax1)
    ax1.set_title('Error Distribution')
    ax1.set_xlabel('Error (Actual - Predicted)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot absolute error histogram
    sns.histplot(df['error'].abs(), kde=True, bins=n_bins, ax=ax2)
    ax2.set_title('Absolute Error Distribution')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Add error statistics
    mean_error = df['error'].mean()
    median_error = df['error'].median()
    std_error = df['error'].std()
    mean_abs_error = df['error'].abs().mean()
    
    stats_text = (
        f"Mean Error: {mean_error:.4f}\n"
        f"Median Error: {median_error:.4f}\n"
        f"Std Deviation: {std_error:.4f}\n"
        f"Mean Abs Error: {mean_abs_error:.4f}"
    )
    
    ax1.text(
        0.95, 0.95, stats_text,
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Error distribution plot saved to {output_path}")
    else:
        plt.show()

def plot_error_by_time(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    group_by: str = 'hour',
    figsize: Tuple[int, int] = (15, 6),
    output_path: Optional[str] = None
) -> None:
    """
    Plot prediction error by time period (hour, day, month, etc.).
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values, datetime index
    actual_column : str, optional
        Name of actual values column, by default 'actual'
    pred_column : str, optional
        Name of predicted values column, by default 'predicted'
    group_by : str, optional
        Time period to group by, by default 'hour'
        Options: 'hour', 'day', 'month', 'weekday', 'season'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 6)
    output_path : str, optional
        Path to save the plot, by default None (display only)
        
    Returns
    -------
    None
    """
    if not isinstance(predictions_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if actual_column not in predictions_df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    logger.info(f"Creating error by {group_by} plot")
    
    # Make a copy of the data to avoid modifying the original
    df = predictions_df.copy()
    
    # Calculate errors
    df['error'] = df[actual_column] - df[pred_column]
    df['abs_error'] = df['error'].abs()
    
    # Drop rows with NaN errors
    df = df.dropna(subset=['error'])
    
    # Create time period grouping
    if group_by == 'hour':
        df['group'] = df.index.hour
        x_label = 'Hour of Day'
    elif group_by == 'day':
        df['group'] = df.index.day
        x_label = 'Day of Month'
    elif group_by == 'month':
        df['group'] = df.index.month
        x_label = 'Month'
    elif group_by == 'weekday':
        df['group'] = df.index.dayofweek
        x_label = 'Day of Week'
        df['group'] = df['group'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
    elif group_by == 'season':
        # Define seasons (Northern Hemisphere)
        month = df.index.month
        df['group'] = np.select(
            [
                (month >= 3) & (month <= 5),    # Spring
                (month >= 6) & (month <= 8),    # Summer
                (month >= 9) & (month <= 11),   # Fall
                (month == 12) | (month <= 2)    # Winter
            ],
            ['Spring', 'Summer', 'Fall', 'Winter']
        )
        x_label = 'Season'
    else:
        raise ValueError(f"Unsupported group_by value: {group_by}")
    
    # Group by time period and calculate error statistics
    grouped = df.groupby('group')
    
    # Calculate metrics for each group
    error_stats = grouped['error'].agg(['mean', 'median', 'std'])
    abs_error_stats = grouped['abs_error'].agg(['mean', 'median', 'std'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot mean error by group
    error_stats['mean'].plot(kind='bar', yerr=error_stats['std'], ax=ax1, alpha=0.7)
    ax1.set_title(f'Mean Error by {group_by.capitalize()}')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Mean Error')
    ax1.grid(True, alpha=0.3)
    
    # Plot mean absolute error by group
    abs_error_stats['mean'].plot(kind='bar', yerr=abs_error_stats['std'], ax=ax2, alpha=0.7)
    ax2.set_title(f'Mean Absolute Error by {group_by.capitalize()}')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Mean Absolute Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Error by {group_by} plot saved to {output_path}")
    else:
        plt.show()

def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    color_map: str = 'viridis'
) -> None:
    """
    Plot feature importance.
    
    Parameters
    ----------
    feature_importance : Dict[str, float]
        Dictionary mapping feature names to importance scores
    top_n : int, optional
        Number of top features to show, by default 20
    title : str, optional
        Plot title, by default 'Feature Importance'
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    color_map : str, optional
        Color map for the bars, by default 'viridis'
        
    Returns
    -------
    None
    """
    logger.info(f"Creating feature importance plot for top {top_n} features")
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Take top N features
    df = df.head(top_n)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create horizontal bar plot
    bars = plt.barh(
        y=df['Feature'],
        width=df['Importance'],
        alpha=0.8,
        color=plt.cm.get_cmap(color_map)(np.linspace(0, 1, len(df)))
    )
    
    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            va='center'
        )
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()

def plot_horizon_metrics(
    evaluation_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None
) -> None:
    """
    Plot forecasting metrics by prediction horizon.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Any]
        Dictionary containing evaluation results with 'by_horizon' key
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
    output_path : str, optional
        Path to save the plot, by default None (display only)
        
    Returns
    -------
    None
    """
    if 'by_horizon' not in evaluation_results or not evaluation_results['by_horizon']:
        logger.warning("No horizon-based metrics found in evaluation results")
        return
    
    logger.info("Creating horizon metrics plot")
    
    # Extract horizons and metrics
    horizons = sorted([int(h) for h in evaluation_results['by_horizon'].keys()])
    
    # Extract metrics for each horizon
    mae_values = [evaluation_results['by_horizon'][str(h)]['MAE'] for h in horizons]
    rmse_values = [evaluation_results['by_horizon'][str(h)]['RMSE'] for h in horizons]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot metrics by horizon
    plt.plot(horizons, mae_values, 'b-o', label='MAE', alpha=0.7)
    plt.plot(horizons, rmse_values, 'r-o', label='RMSE', alpha=0.7)
    
    # Add title and labels
    plt.title('Forecast Accuracy by Prediction Horizon')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Error')
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line for overall MAE
    if 'overall' in evaluation_results and 'MAE' in evaluation_results['overall']:
        overall_mae = evaluation_results['overall']['MAE']
        plt.axhline(y=overall_mae, color='b', linestyle='--', alpha=0.5)
        plt.text(horizons[-1], overall_mae, f'Overall MAE: {overall_mae:.4f}', 
                 verticalalignment='bottom', horizontalalignment='right')
    
    # Add horizontal line for overall RMSE
    if 'overall' in evaluation_results and 'RMSE' in evaluation_results['overall']:
        overall_rmse = evaluation_results['overall']['RMSE']
        plt.axhline(y=overall_rmse, color='r', linestyle='--', alpha=0.5)
        plt.text(horizons[-1], overall_rmse, f'Overall RMSE: {overall_rmse:.4f}', 
                 verticalalignment='top', horizontalalignment='right')
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Horizon metrics plot saved to {output_path}")
    else:
        plt.show()

def plot_temperature_heatmap(
    df: pd.DataFrame,
    temperature_column: str = 'OT',
    period: str = 'daily',
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    cmap: str = 'inferno'
) -> None:
    """
    Create a heatmap to visualize temperature patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with temperature data, datetime index
    temperature_column : str, optional
        Name of temperature column, by default 'OT'
    period : str, optional
        Period to analyze, by default 'daily'
        Options: 'daily', 'weekly', 'monthly', 'yearly'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    cmap : str, optional
        Colormap for heatmap, by default 'inferno'
        
    Returns
    -------
    None
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if temperature_column not in df.columns:
        raise ValueError(f"Temperature column '{temperature_column}' not found in DataFrame")
    
    logger.info(f"Creating {period} temperature heatmap")
    
    # Make a copy of the data to avoid modifying the original
    data = df.copy()
    
    # Create pivot table based on period
    if period == 'daily':
        # Create hour of day vs. day heatmap
        data['hour'] = data.index.hour
        data['date'] = data.index.date
        
        pivot_df = data.pivot_table(
            values=temperature_column,
            index='hour',
            columns='date',
            aggfunc='mean'
        )
        
        title = 'Daily Temperature Pattern (Hour vs. Day)'
        x_label = 'Day'
        y_label = 'Hour of Day'
    
    elif period == 'weekly':
        # Create hour of day vs. day of week heatmap
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        
        pivot_df = data.pivot_table(
            values=temperature_column,
            index='hour',
            columns='dayofweek',
            aggfunc='mean'
        )
        
        # Map day numbers to day names
        day_mapping = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        pivot_df.columns = [day_mapping[d] for d in pivot_df.columns]
        
        title = 'Weekly Temperature Pattern (Hour vs. Day of Week)'
        x_label = 'Day of Week'
        y_label = 'Hour of Day'
    
    elif period == 'monthly':
        # Create day vs. month heatmap
        data['day'] = data.index.day
        data['month'] = data.index.month
        
        pivot_df = data.pivot_table(
            values=temperature_column,
            index='day',
            columns='month',
            aggfunc='mean'
        )
        
        # Map month numbers to month names
        month_mapping = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot_df.columns = [month_mapping[m] for m in pivot_df.columns]
        
        title = 'Monthly Temperature Pattern (Day vs. Month)'
        x_label = 'Month'
        y_label = 'Day of Month'
    
    elif period == 'yearly':
        # Create month vs. year heatmap
        data['month'] = data.index.month
        data['year'] = data.index.year
        
        pivot_df = data.pivot_table(
            values=temperature_column,
            index='month',
            columns='year',
            aggfunc='mean'
        )
        
        # Map month numbers to month names
        month_mapping = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot_df.index = [month_mapping[m] for m in pivot_df.index]
        
        title = 'Yearly Temperature Pattern (Month vs. Year)'
        x_label = 'Year'
        y_label = 'Month'
    
    else:
        raise ValueError(f"Unsupported period: {period}")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        cmap=cmap,
        annot=False,
        fmt='.1f',
        cbar_kws={'label': 'Temperature (°C)'}
    )
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Temperature heatmap saved to {output_path}")
    else:
        plt.show()

def plot_error_vs_value(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    bins: int = 20
) -> None:
    """
    Plot prediction error vs actual value to analyze model performance across temperature ranges.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    actual_column : str, optional
        Name of actual values column, by default 'actual'
    pred_column : str, optional
        Name of predicted values column, by default 'predicted'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    bins : int, optional
        Number of temperature bins for aggregation, by default 20
        
    Returns
    -------
    None
    """
    if actual_column not in predictions_df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    logger.info("Creating error vs value plot")
    
    # Make a copy of the data to avoid modifying the original
    df = predictions_df.copy()
    
    # Calculate errors
    df['error'] = df[actual_column] - df[pred_column]
    df['abs_error'] = df['error'].abs()
    df['pct_error'] = (df['error'] / df[actual_column]) * 100
    
    # Drop rows with NaN values
    df = df.dropna(subset=[actual_column, pred_column, 'error', 'abs_error'])
    
    # Create temperature bins
    df['temp_bin'] = pd.cut(
        df[actual_column],
        bins=bins,
        labels=False
    )
    
    # Group by temperature bin and calculate statistics
    bin_stats = df.groupby('temp_bin').agg({
        actual_column: 'mean',
        'error': ['mean', 'std'],
        'abs_error': 'mean',
        'pct_error': 'mean'
    })
    
    # Flatten column names
    bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns.values]
    
    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Mean error by temperature
    ax1.errorbar(
        bin_stats[f'{actual_column}_mean'],
        bin_stats['error_mean'],
        yerr=bin_stats['error_std'],
        fmt='o-',
        alpha=0.7
    )
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Mean Error by Temperature')
    ax1.set_ylabel('Mean Error')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean absolute error by temperature
    ax2.plot(
        bin_stats[f'{actual_column}_mean'],
        bin_stats['abs_error_mean'],
        'o-',
        alpha=0.7
    )
    ax2.set_title('Mean Absolute Error by Temperature')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean percentage error by temperature
    ax3.plot(
        bin_stats[f'{actual_column}_mean'],
        bin_stats['pct_error_mean'].abs(),
        'o-',
        alpha=0.7
    )
    ax3.set_title('Mean Absolute Percentage Error by Temperature')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('MAPE (%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Error vs value plot saved to {output_path}")
    else:
        plt.show()

def compare_models(
    predictions_dict: Dict[str, pd.DataFrame],
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    metrics: List[str] = ['MAE', 'RMSE', 'R2']
) -> None:
    """
    Compare performance of multiple models.
    
    Parameters
    ----------
    predictions_dict : Dict[str, pd.DataFrame]
        Dictionary mapping model names to prediction DataFrames
    actual_column : str, optional
        Name of actual values column, by default 'actual'
    pred_column : str, optional
        Name of predicted values column, by default 'predicted'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
    output_path : str, optional
        Path to save the plot, by default None (display only)
    metrics : List[str], optional
        List of metrics to compare, by default ['MAE', 'RMSE', 'R2']
        
    Returns
    -------
    None
    """
    logger.info(f"Comparing {len(predictions_dict)} models")
    
    # Calculate metrics for each model
    model_metrics = {}
    
    for model_name, df in predictions_dict.items():
        if actual_column not in df.columns:
            logger.warning(f"Actual column '{actual_column}' not found for model '{model_name}', skipping")
            continue
        
        if pred_column not in df.columns:
            logger.warning(f"Prediction column '{pred_column}' not found for model '{model_name}', skipping")
            continue
        
        # Drop rows with NaN values
        valid_df = df.dropna(subset=[actual_column, pred_column])
        
        if len(valid_df) == 0:
            logger.warning(f"No valid data points for model '{model_name}', skipping")
            continue
        
        # Calculate metrics
        y_true = valid_df[actual_column].values
        y_pred = valid_df[pred_column].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE with handling for zero values
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
        
        model_metrics[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    if not model_metrics:
        logger.warning("No valid models for comparison")
        return
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(model_metrics).T
    
    # Filter to requested metrics
    metrics_df = metrics_df[[m for m in metrics if m in metrics_df.columns]]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create subplots for each metric
    n_metrics = len(metrics_df.columns)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    # Handle case with only one metric
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics_df.columns):
        ax = axes[i]
        
        # Sort models by metric (ascending for error metrics, descending for R2)
        if metric == 'R2':
            sorted_models = metrics_df.sort_values(metric, ascending=False).index
        else:
            sorted_models = metrics_df.sort_values(metric, ascending=True).index
        
        # Create bar plot
        ax.bar(
            np.arange(len(sorted_models)),
            [metrics_df.loc[model, metric] for model in sorted_models],
            alpha=0.7
        )
        
        # Add values on top of bars
        for j, model in enumerate(sorted_models):
            value = metrics_df.loc[model, metric]
            ax.text(
                j, value,
                f'{value:.4f}',
                ha='center',
                va='bottom'
            )
        
        # Set labels
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xticks(np.arange(len(sorted_models)))
        ax.set_xticklabels(sorted_models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Model comparison plot saved to {output_path}")
    else:
        plt.show()

def create_interactive_dashboard(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    sequence_id_column: str = 'sequence_id',
    title: str = 'Forecasting Dashboard',
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive dashboard for exploring forecasting results.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    actual_column : str, optional
        Name of actual values column, by default 'actual'
    pred_column : str, optional
        Name of predicted values column, by default 'predicted'
    sequence_id_column : str, optional
        Name of sequence ID column, by default 'sequence_id'
    title : str, optional
        Dashboard title, by default 'Forecasting Dashboard'
    output_path : str, optional
        Path to save the HTML dashboard, by default None
        
    Returns
    -------
    go.Figure
        Plotly figure object for the dashboard
    """
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    # Check if we have actual values
    has_actual = actual_column in predictions_df.columns
    
    # Make a copy of the data to avoid modifying the original
    df = predictions_df.copy()
    
    if sequence_id_column not in df.columns:
        logger.warning(f"Sequence ID column '{sequence_id_column}' not found, creating artificial sequences")
        # Create artificial sequence IDs based on date
        if isinstance(df.index, pd.DatetimeIndex):
            df[sequence_id_column] = df.index.date.astype(str)
        else:
            # Just use row numbers if no proper index
            df[sequence_id_column] = np.arange(len(df))
    
    # Calculate error if we have actual values
    if has_actual:
        df['error'] = df[actual_column] - df[pred_column]
        df['abs_error'] = df['error'].abs()
    
    # Get unique sequence IDs
    sequence_ids = df[sequence_id_column].unique()
    
    # Create interactive dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction vs Actual',
            'Error Distribution' if has_actual else 'Predictions Over Time',
            'Error by Time' if has_actual else 'Predictions by Sequence',
            'Performance Metrics' if has_actual else 'Sequence Summary'
        ),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    # Main plot: Predictions vs Actual (or just predictions if no actual data)
    if has_actual:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[actual_column],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[pred_column],
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Second plot: Error Distribution or Predictions Over Time
    if has_actual:
        fig.add_trace(
            go.Histogram(
                x=df['error'],
                nbinsx=50,
                name='Error Distribution',
                marker_color='indianred'
            ),
            row=2, col=1
        )
    else:
        # Create a heatmap of predictions by time of day and day
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day'] = df.index.date
            
            # Group by hour and day
            pivot_data = df.pivot_table(
                values=pred_column,
                index='hour',
                columns='day',
                aggfunc='mean'
            )
            
            # Convert to format for heatmap
            hours = pivot_data.index
            days = pivot_data.columns
            z_data = pivot_data.values
            
            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=days,
                    y=hours,
                    colorscale='Viridis',
                    name='Predictions by Time'
                ),
                row=2, col=1
            )
        else:
            # Just show predictions over time if no datetime index
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(df)),
                    y=df[pred_column],
                    mode='lines',
                    name='Predictions',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
    
    # Third plot: Error by Time or Predictions by Sequence
    if has_actual:
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            hourly_errors = df.groupby('hour')['abs_error'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=hourly_errors.index,
                    y=hourly_errors.values,
                    name='Error by Hour',
                    marker_color='gold'
                ),
                row=2, col=2
            )
        else:
            # If no datetime index, show error by sequence
            sequence_errors = df.groupby(sequence_id_column)['abs_error'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=sequence_errors.index,
                    y=sequence_errors.values,
                    name='Error by Sequence',
                    marker_color='gold'
                ),
                row=2, col=2
            )
    else:
        # Show predictions by sequence
        sequence_means = df.groupby(sequence_id_column)[pred_column].mean()
        
        fig.add_trace(
            go.Bar(
                x=sequence_means.index,
                y=sequence_means.values,
                name='Mean Prediction by Sequence',
                marker_color='purple'
            ),
            row=2, col=2
        )
    
    # Add performance metrics or sequence summary as text annotation
    if has_actual:
        mae = df['abs_error'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        
        # Calculate MAPE with handling for zero values
        mape = np.mean(np.abs(df['error'] / np.maximum(np.abs(df[actual_column]), 1e-10))) * 100
        
        metrics_text = (
            f"<b>Model Performance Metrics</b><br><br>"
            f"MAE: {mae:.4f}<br>"
            f"RMSE: {rmse:.4f}<br>"
            f"MAPE: {mape:.2f}%<br><br>"
            f"Total Points: {len(df)}<br>"
            f"Sequences: {len(sequence_ids)}"
        )
    else:
        # Just show summary statistics for predictions
        pred_mean = df[pred_column].mean()
        pred_min = df[pred_column].min()
        pred_max = df[pred_column].max()
        pred_std = df[pred_column].std()
        
        metrics_text = (
            f"<b>Prediction Summary</b><br><br>"
            f"Mean: {pred_mean:.4f}<br>"
            f"Min: {pred_min:.4f}<br>"
            f"Max: {pred_max:.4f}<br>"
            f"Std Dev: {pred_std:.4f}<br><br>"
            f"Total Points: {len(df)}<br>"
            f"Sequences: {len(sequence_ids)}"
        )
    
    # Add the metrics annotation
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.5,
        text=metrics_text,
        showarrow=False,
        font=dict(
            family="Arial",
            size=14,
            color="black"
        ),
        align="center",
        bordercolor="black",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        opacity=0.8,
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis titles
    fig.update_xaxes(title_text="Time", row=1, col=1)
    
    if has_actual:
        fig.update_xaxes(title_text="Error (Actual - Predicted)", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day" if isinstance(df.index, pd.DatetimeIndex) else "Sequence ID", row=2, col=2)
    else:
        fig.update_xaxes(title_text="Day" if isinstance(df.index, pd.DatetimeIndex) else "Time", row=2, col=1)
        fig.update_xaxes(title_text="Sequence ID", row=2, col=2)
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Temperature", row=1, col=1)
    
    if has_actual:
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error", row=2, col=2)
    else:
        fig.update_yaxes(title_text="Hour of Day" if isinstance(df.index, pd.DatetimeIndex) else "Value", row=2, col=1)
        fig.update_yaxes(title_text="Mean Prediction", row=2, col=2)
    
    # Save dashboard if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        logger.info(f"Interactive dashboard saved to {output_path}")
    
    return fig

def plot_load_vs_temperature(
    df: pd.DataFrame,
    load_columns: List[str],
    temperature_column: str = 'OT',
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None
) -> None:
    """
    Analyze relationship between load values and oil temperature.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with load and temperature data
    load_columns : List[str]
        List of load columns to analyze
    temperature_column : str, optional
        Name of temperature column, by default 'OT'
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
    output_path : str, optional
        Path to save the plot, by default None (display only)
        
    Returns
    -------
    None
    """
    if temperature_column not in df.columns:
        raise ValueError(f"Temperature column '{temperature_column}' not found in DataFrame")
    
    available_load_columns = [col for col in load_columns if col in df.columns]
    
    if not available_load_columns:
        raise ValueError(f"None of the specified load columns found in DataFrame")
    
    logger.info(f"Analyzing relationship between {len(available_load_columns)} load columns and temperature")
    
    # Create figure with subplots (1 row per load column)
    n_cols = 3  # Scatter, Time Series, Box Plot
    n_rows = len(available_load_columns)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single row case
    if n_rows == 1:
        axes = [axes]
    
    # Analyze each load column
    for i, load_col in enumerate(available_load_columns):
        # Make sure we're working with clean data
        valid_data = df[[load_col, temperature_column]].dropna()
        
        # 1. Scatter plot with regression line
        ax1 = axes[i][0]
        sns.regplot(
            x=load_col,
            y=temperature_column,
            data=valid_data,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'},
            ax=ax1
        )
        ax1.set_title(f"{load_col} vs. {temperature_column}")
        ax1.set_xlabel(load_col)
        ax1.set_ylabel(temperature_column)
        
        # Calculate correlation
        corr = valid_data[load_col].corr(valid_data[temperature_column])
        ax1.text(
            0.05, 0.95,
            f"Correlation: {corr:.4f}",
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # 2. Time series of both variables
        if isinstance(valid_data.index, pd.DatetimeIndex):
            ax2 = axes[i][1]
            
            # Create twin axis for different scales
            ax2_twin = ax2.twinx()
            
            # Plot load on primary axis
            ax2.plot(valid_data.index, valid_data[load_col], 'b-', alpha=0.7, label=load_col)
            ax2.set_ylabel(load_col, color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Plot temperature on secondary axis
            ax2_twin.plot(valid_data.index, valid_data[temperature_column], 'r-', alpha=0.7, label=temperature_column)
            ax2_twin.set_ylabel(temperature_column, color='r')
            ax2_twin.tick_params(axis='y', labelcolor='r')
            
            ax2.set_title(f"{load_col} and {temperature_column} Over Time")
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        else:
            ax2 = axes[i][1]
            ax2.text(
                0.5, 0.5,
                "Time series plot requires DatetimeIndex",
                ha='center', va='center',
                transform=ax2.transAxes
            )
        
        # 3. Box plot of temperature by load bins
        ax3 = axes[i][2]
        
        # Create load bins
        n_bins = 5
        valid_data['load_bin'] = pd.cut(
            valid_data[load_col],
            bins=n_bins,
            labels=[f'Bin {j+1}' for j in range(n_bins)]
        )
        
        # Box plot of temperature by load bin
        sns.boxplot(
            x='load_bin',
            y=temperature_column,
            data=valid_data,
            ax=ax3
        )
        ax3.set_title(f"{temperature_column} by {load_col} Bins")
        ax3.set_xlabel(f"{load_col} Bins")
        ax3.set_ylabel(temperature_column)
        
        # Rotate x-axis labels for better readability
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Load vs temperature plot saved to {output_path}")
    else:
        plt.show()

def plot_extreme_temperatures_analysis(
    df: pd.DataFrame,
    temperature_column: str = 'OT',
    threshold_percentile: float = 95,
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze and visualize extreme temperature events.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with temperature data, datetime index
    temperature_column : str, optional
        Name of temperature column, by default 'OT'
    threshold_percentile : float, optional
        Percentile to define extreme temperatures, by default 95
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
    output_path : str, optional
        Path to save the plot, by default None (display only)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with extreme temperature analysis results
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if temperature_column not in df.columns:
        raise ValueError(f"Temperature column '{temperature_column}' not found in DataFrame")
    
    logger.info(f"Analyzing extreme temperatures (>{threshold_percentile}th percentile)")
    
    # Make a copy of the data to avoid modifying the original
    data = df.copy()
    
    # Calculate threshold for extreme temperatures
    threshold = data[temperature_column].quantile(threshold_percentile / 100)
    
    # Identify extreme temperature events
    data['is_extreme'] = data[temperature_column] >= threshold
    
    # Count extreme events
    extreme_count = data['is_extreme'].sum()
    total_count = len(data)
    extreme_percentage = (extreme_count / total_count) * 100
    
    logger.info(f"Found {extreme_count} extreme temperature events (≥{threshold:.2f}°C), "
                f"{extreme_percentage:.2f}% of all data points")
    
    # Create monthly statistics
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    
    # Group by month and calculate extreme event frequency
    monthly_extremes = data.groupby('month')['is_extreme'].mean() * 100
    
    # Group by hour and calculate extreme event frequency
    hourly_extremes = data.groupby('hour')['is_extreme'].mean() * 100
    
    # Group by day of week and calculate extreme event frequency
    day_mapping = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    data['day_name'] = data['day_of_week'].map(day_mapping)
    daily_extremes = data.groupby('day_name')['is_extreme'].mean() * 100
    
    # Create extreme temperature duration analysis
    # Find consecutive runs of extreme temperatures
    data['extreme_start'] = data['is_extreme'] & ~data['is_extreme'].shift(1).fillna(False)
    data['extreme_end'] = data['is_extreme'] & ~data['is_extreme'].shift(-1).fillna(False)
    
    # Extract durations of extreme temperature events
    extreme_starts = data[data['extreme_start']].index
    extreme_ends = data[data['extreme_end']].index
    
    durations = []
    for start, end in zip(extreme_starts, extreme_ends):
        duration = (end - start).total_seconds() / 3600  # Convert to hours
        durations.append(duration)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Extreme temperature events over time
    ax1.scatter(
        data.index[data['is_extreme']],
        data.loc[data['is_extreme'], temperature_column],
        color='red',
        alpha=0.7,
        label=f'Extreme Temperatures (≥{threshold:.2f}°C)'
    )
    ax1.plot(
        data.index,
        data[temperature_column],
        'b-',
        alpha=0.3,
        label='All Temperatures'
    )
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({threshold:.2f}°C)')
    ax1.set_title('Extreme Temperature Events Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Plot 2: Monthly distribution of extreme temperatures
    month_mapping = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_extremes.index = [month_mapping[m] for m in monthly_extremes.index]
    
    ax2.bar(
        monthly_extremes.index,
        monthly_extremes.values,
        alpha=0.7,
        color='orange'
    )
    ax2.set_title('Monthly Distribution of Extreme Temperatures')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Percentage of Time Points (%)')
    
    # Add percentage values on bars
    for i, v in enumerate(monthly_extremes.values):
        ax2.text(
            i, v + 0.5,
            f'{v:.1f}%',
            ha='center'
        )
    
    # Plot 3: Hourly distribution of extreme temperatures
    ax3.bar(
        hourly_extremes.index,
        hourly_extremes.values,
        alpha=0.7,
        color='green'
    )
    ax3.set_title('Hourly Distribution of Extreme Temperatures')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Percentage of Time Points (%)')
    ax3.set_xticks(range(0, 24, 2))
    
    # Add percentage values on bars
    for i, v in enumerate(hourly_extremes.values):
        if v > 2:  # Only show labels for bars with sufficient height
            ax3.text(
                i, v + 0.5,
                f'{v:.1f}%',
                ha='center'
            )
    
    # Plot 4: Distribution of extreme temperature event durations
    if durations:
        ax4.hist(
            durations,
            bins=20,
            alpha=0.7,
            color='purple'
        )
        ax4.set_title('Duration of Extreme Temperature Events')
        ax4.set_xlabel('Duration (hours)')
        ax4.set_ylabel('Frequency')
        
        # Add statistics
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        max_duration = np.max(durations)
        
        stats_text = (
            f"Mean: {mean_duration:.2f} hours\n"
            f"Median: {median_duration:.2f} hours\n"
            f"Max: {max_duration:.2f} hours\n"
            f"Events: {len(durations)}"
        )
        
        ax4.text(
            0.95, 0.95,
            stats_text,
            transform=ax4.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    else:
        ax4.text(
            0.5, 0.5,
            "No extreme temperature events with duration data",
            ha='center', va='center',
            transform=ax4.transAxes
        )
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Extreme temperatures analysis plot saved to {output_path}")
    else:
        plt.show()
    
    # Return analysis results
    results = {
        'threshold': float(threshold),
        'extreme_count': int(extreme_count),
        'extreme_percentage': float(extreme_percentage),
        'monthly_distribution': monthly_extremes.to_dict(),
        'hourly_distribution': hourly_extremes.to_dict(),
        'day_of_week_distribution': daily_extremes.to_dict()
    }
    
    if durations:
        results['durations'] = {
            'mean': float(mean_duration),
            'median': float(median_duration),
            'max': float(max_duration),
            'event_count': int(len(durations))
        }
    
    return results

if __name__ == "__main__":
    # Example usage
    project_dir = Path(__file__).resolve().parents[2]
    
    # Define paths
    data_dir = project_dir / "data"
    figures_dir = project_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Example: Load and plot predictions
    predictions_path = data_dir / "predictions" / "test_predictions.csv"
    
    try:
        # Load predictions
        print(f"Loading predictions from {predictions_path}")
        predictions_df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
        
        # Create basic prediction vs actual plot
        plot_prediction_vs_actual(
            predictions_df=predictions_df,
            actual_column='actual',
            pred_column='predicted',
            output_path=str(figures_dir / "prediction_vs_actual.png")
        )
        
        # Create error distribution plot
        plot_error_distribution(
            predictions_df=predictions_df,
            actual_column='actual',
            pred_column='predicted',
            output_path=str(figures_dir / "error_distribution.png")
        )
        
        # Create error by time plot
        plot_error_by_time(
            predictions_df=predictions_df,
            actual_column='actual',
            pred_column='predicted',
            group_by='hour',
            output_path=str(figures_dir / "error_by_hour.png")
        )
        
        # Create prediction sequences plot
        plot_prediction_sequences(
            predictions_df=predictions_df,
            num_sequences=5,
            actual_column='actual',
            pred_column='predicted',
            output_path=str(figures_dir / "prediction_sequences.png")
        )
        
        # Create error vs temperature plot
        plot_error_vs_value(
            predictions_df=predictions_df,
            actual_column='actual',
            pred_column='predicted',
            output_path=str(figures_dir / "error_vs_temperature.png")
        )
        
        # Create interactive dashboard
        dashboard_path = project_dir / "reports" / "dashboard.html"
        create_interactive_dashboard(
            predictions_df=predictions_df,
            actual_column='actual',
            pred_column='predicted',
            output_path=str(dashboard_path)
        )
        
        print(f"Visualizations saved to {figures_dir}")
        print(f"Interactive dashboard saved to {dashboard_path}")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")