"""
Functions for evaluating forecasting models for transformer oil temperature.

This module contains functions to:
1. Calculate common evaluation metrics (MAE, RMSE, MAPE)
2. Evaluate model performance across different forecast horizons
3. Plot predictions against actual values
4. Analyze error patterns and potential model improvements
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: str = 'uniform_average'
) -> Dict[str, float]:
    """
    Calculate common evaluation metrics for forecasting.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    multioutput : str, optional
        How to handle multi-output metrics, by default 'uniform_average'
        Options: 'uniform_average', 'raw_values'
        
    Returns
    -------
    Dict[str, float]
        Dictionary with evaluation metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check for NaN or infinity
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        logger.warning("NaN values found in inputs, metrics may be unreliable")
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    
    # Root Mean Squared Error (RMSE)
    mse = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    # Handle division by zero
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
        logger.warning("Cannot calculate MAPE due to zero values in y_true")
    
    # R-squared
    r2 = r2_score(y_true, y_pred, multioutput=multioutput)
    
    # Add metrics to dictionary
    if multioutput == 'raw_values':
        metrics['MAE'] = mae.tolist() if isinstance(mae, np.ndarray) else mae
        metrics['RMSE'] = rmse.tolist() if isinstance(rmse, np.ndarray) else rmse
        metrics['R2'] = r2.tolist() if isinstance(r2, np.ndarray) else r2
        metrics['MAPE'] = mape  # Scalar value even with raw_values
    else:
        metrics['MAE'] = float(mae)
        metrics['RMSE'] = float(rmse)
        metrics['MAPE'] = float(mape)
        metrics['R2'] = float(r2)
    
    return metrics

def evaluate_model(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    by_horizon: bool = True,
    by_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance on predictions.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    actual_column : str, optional
        Name of column with actual values, by default 'actual'
    pred_column : str, optional
        Name of column with predicted values, by default 'predicted'
    by_horizon : bool, optional
        Whether to calculate metrics by forecast horizon, by default True
    by_time : str, optional
        Time period to group by for evaluation, by default None
        Options: 'hour', 'day', 'month', 'season', etc.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating model on {len(predictions_df)} predictions")
    
    # Check if actual values are available
    if actual_column not in predictions_df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(
        predictions_df[actual_column].values,
        predictions_df[pred_column].values
    )
    
    logger.info(f"Overall metrics: MAE={overall_metrics['MAE']:.4f}, RMSE={overall_metrics['RMSE']:.4f}, MAPE={overall_metrics['MAPE']:.2f}%")
    
    results = {
        'overall': overall_metrics,
        'by_horizon': {},
        'by_time': {}
    }
    
    # Calculate metrics by forecast horizon
    if by_horizon and 'sequence_id' in predictions_df.columns:
        # Group by sequence_id and calculate position in sequence
        grouped = predictions_df.groupby('sequence_id')
        
        # Calculate metrics for each horizon
        horizon_metrics = {}
        
        for seq_id, group in grouped:
            # Sort by index to ensure correct order
            group = group.sort_index()
            
            # Calculate position in forecast (horizon)
            group['horizon'] = np.arange(len(group))
            
            for horizon, horizon_group in group.groupby('horizon'):
                if horizon not in horizon_metrics:
                    horizon_metrics[horizon] = {
                        'actual': [],
                        'predicted': []
                    }
                
                horizon_metrics[horizon]['actual'].extend(horizon_group[actual_column].values)
                horizon_metrics[horizon]['predicted'].extend(horizon_group[pred_column].values)
        
        # Calculate metrics for each horizon
        for horizon, data in horizon_metrics.items():
            metrics = calculate_metrics(
                np.array(data['actual']),
                np.array(data['predicted'])
            )
            results['by_horizon'][horizon] = metrics
        
        logger.info(f"Calculated metrics by horizon for {len(horizon_metrics)} horizons")
    
    # Calculate metrics by time period
    if by_time:
        # Create time features
        if isinstance(predictions_df.index, pd.DatetimeIndex):
            if by_time == 'hour':
                predictions_df['time_group'] = predictions_df.index.hour
            elif by_time == 'day':
                predictions_df['time_group'] = predictions_df.index.day
            elif by_time == 'month':
                predictions_df['time_group'] = predictions_df.index.month
            elif by_time == 'season':
                # Define seasons (Northern Hemisphere)
                month = predictions_df.index.month
                predictions_df['time_group'] = np.select(
                    [
                        (month >= 3) & (month <= 5),    # Spring
                        (month >= 6) & (month <= 8),    # Summer
                        (month >= 9) & (month <= 11),   # Fall
                        (month == 12) | (month <= 2)    # Winter
                    ],
                    ['Spring', 'Summer', 'Fall', 'Winter']
                )
            elif by_time == 'weekday':
                predictions_df['time_group'] = predictions_df.index.dayofweek
            else:
                logger.warning(f"Unsupported time grouping: {by_time}")
                return results
            
            # Calculate metrics for each time group
            for time_group, group in predictions_df.groupby('time_group'):
                metrics = calculate_metrics(
                    group[actual_column].values,
                    group[pred_column].values
                )
                results['by_time'][time_group] = metrics
            
            logger.info(f"Calculated metrics by {by_time} for {len(results['by_time'])} groups")
        else:
            logger.warning("DataFrame index is not DatetimeIndex, cannot group by time")
    
    return results

def plot_predictions(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    sequence_id_column: str = 'sequence_id',
    num_sequences: int = 5,
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None
) -> None:
    """
    Plot predictions against actual values for selected sequences.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    actual_column : str, optional
        Name of column with actual values, by default 'actual'
    pred_column : str, optional
        Name of column with predicted values, by default 'predicted'
    sequence_id_column : str, optional
        Name of column with sequence IDs, by default 'sequence_id'
    num_sequences : int, optional
        Number of sequences to plot, by default 5
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 10)
    output_path : str, optional
        Path to save the plot, by default None (display only)
        
    Returns
    -------
    None
    """
    # Check if required columns exist
    if actual_column not in predictions_df.columns:
        logger.warning(f"Actual column '{actual_column}' not found in DataFrame")
        has_actual = False
    else:
        has_actual = True
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    if sequence_id_column not in predictions_df.columns:
        raise ValueError(f"Sequence ID column '{sequence_id_column}' not found in DataFrame")
    
    # Get unique sequence IDs
    sequence_ids = predictions_df[sequence_id_column].unique()
    
    # Limit number of sequences to plot
    num_sequences = min(num_sequences, len(sequence_ids))
    selected_sequences = np.random.choice(sequence_ids, size=num_sequences, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(num_sequences, 1, figsize=figsize, sharex=True)
    if num_sequences == 1:
        axes = [axes]
    
    # Plot each selected sequence
    for i, seq_id in enumerate(selected_sequences):
        seq_data = predictions_df[predictions_df[sequence_id_column] == seq_id].sort_index()
        
        ax = axes[i]
        ax.plot(seq_data.index, seq_data[pred_column], 'r-', label='Predicted')
        
        if has_actual:
            ax.plot(seq_data.index, seq_data[actual_column], 'b-', label='Actual')
        
        # Calculate metrics for this sequence if actual values exist
        if has_actual:
            metrics = calculate_metrics(
                seq_data[actual_column].values,
                seq_data[pred_column].values
            )
            title = f"Sequence {seq_id}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%"
        else:
            title = f"Sequence {seq_id}"
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    
    # Set common labels
    fig.text(0.5, 0.04, 'Time', ha='center', va='center')
    fig.text(0.06, 0.5, 'Oil Temperature (°C)', ha='center', va='center', rotation='vertical')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1)
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Predictions plot saved to {output_path}")
    else:
        plt.show()

def plot_metrics_by_horizon(
    evaluation_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[str] = None
) -> None:
    """
    Plot metrics by forecast horizon.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Any]
        Evaluation results from evaluate_model function
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6)
    output_path : str, optional
        Path to save the plot, by default None (display only)
        
    Returns
    -------
    None
    """
    if 'by_horizon' not in evaluation_results or not evaluation_results['by_horizon']:
        logger.warning("No horizon-based metrics found in evaluation results")
        return
    
    # Extract horizons and metrics
    horizons = sorted(evaluation_results['by_horizon'].keys())
    
    mae_values = [evaluation_results['by_horizon'][h]['MAE'] for h in horizons]
    rmse_values = [evaluation_results['by_horizon'][h]['RMSE'] for h in horizons]
    mape_values = [evaluation_results['by_horizon'][h]['MAPE'] for h in horizons]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot MAE by horizon
    ax1.plot(horizons, mae_values, 'o-')
    ax1.set_title('MAE by Forecast Horizon')
    ax1.set_xlabel('Forecast Horizon (hours)')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.grid(True)
    
    # Plot RMSE by horizon
    ax2.plot(horizons, rmse_values, 'o-')
    ax2.set_title('RMSE by Forecast Horizon')
    ax2.set_xlabel('Forecast Horizon (hours)')
    ax2.set_ylabel('Root Mean Squared Error')
    ax2.grid(True)
    
    # Plot MAPE by horizon
    ax3.plot(horizons, mape_values, 'o-')
    ax3.set_title('MAPE by Forecast Horizon')
    ax3.set_xlabel('Forecast Horizon (hours)')
    ax3.set_ylabel('Mean Absolute Percentage Error (%)')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Horizon metrics plot saved to {output_path}")
    else:
        plt.show()

def plot_residuals(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None
) -> None:
    """
    Plot residuals analysis for model diagnostics.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    actual_column : str, optional
        Name of column with actual values, by default 'actual'
    pred_column : str, optional
        Name of column with predicted values, by default 'predicted'
    figsize : Tuple[int, int], optional
        Figure size, by default (16, 8)
    output_path : str, optional
        Path to save the plot, by default None (display only)
        
    Returns
    -------
    None
    """
    # Check if required columns exist
    if actual_column not in predictions_df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    # Calculate residuals
    predictions_df = predictions_df.copy()
    predictions_df['residuals'] = predictions_df[actual_column] - predictions_df[pred_column]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Residuals vs. predicted
    ax1.scatter(predictions_df[pred_column], predictions_df['residuals'], alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='-')
    ax1.set_title('Residuals vs. Predicted Values')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.grid(True)
    
    # Plot 2: Residuals distribution
    sns.histplot(predictions_df['residuals'], kde=True, ax=ax2)
    ax2.set_title('Residuals Distribution')
    ax2.set_xlabel('Residuals')
    ax2.grid(True)
    
    # Plot 3: Actual vs. Predicted
    ax3.scatter(predictions_df[actual_column], predictions_df[pred_column], alpha=0.5)
    min_val = min(predictions_df[actual_column].min(), predictions_df[pred_column].min())
    max_val = max(predictions_df[actual_column].max(), predictions_df[pred_column].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r-')
    ax3.set_title('Actual vs. Predicted Values')
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Residuals plot saved to {output_path}")
    else:
        plt.show()

def save_evaluation_results(
    evaluation_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save evaluation results to a JSON file.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Any]
        Evaluation results from evaluate_model function
    output_path : str
        Path to save the results
        
    Returns
    -------
    None
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert any numpy types to native Python types for JSON serialization
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
    
    serializable_results = convert_to_serializable(evaluation_results)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {output_path}")

def analyze_extreme_cases(
    predictions_df: pd.DataFrame,
    actual_column: str = 'actual',
    pred_column: str = 'predicted',
    percentile_threshold: float = 90,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze model performance on extreme cases (high oil temperatures).
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and actual values
    actual_column : str, optional
        Name of column with actual values, by default 'actual'
    pred_column : str, optional
        Name of column with predicted values, by default 'predicted'
    percentile_threshold : float, optional
        Percentile to define extreme cases, by default 90
    output_path : str, optional
        Path to save analysis results, by default None
        
    Returns
    -------
    Dict[str, Any]
        Analysis of extreme cases
    """
    # Check if required columns exist
    if actual_column not in predictions_df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in DataFrame")
    
    if pred_column not in predictions_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in DataFrame")
    
    # Define extreme cases
    threshold = np.percentile(predictions_df[actual_column], percentile_threshold)
    extreme_cases = predictions_df[predictions_df[actual_column] >= threshold].copy()
    
    logger.info(f"Analyzing {len(extreme_cases)} extreme cases (>= {threshold:.2f}°C)")
    
    # Calculate metrics for extreme cases
    extreme_metrics = calculate_metrics(
        extreme_cases[actual_column].values,
        extreme_cases[pred_column].values
    )
    
    # Calculate error statistics
    extreme_cases['error'] = extreme_cases[actual_column] - extreme_cases[pred_column]
    extreme_cases['abs_error'] = np.abs(extreme_cases['error'])
    extreme_cases['pct_error'] = (extreme_cases['error'] / extreme_cases[actual_column]) * 100
    
    error_stats = {
        'mean_error': float(extreme_cases['error'].mean()),
        'median_error': float(extreme_cases['error'].median()),
        'std_error': float(extreme_cases['error'].std()),
        'max_underestimation': float(extreme_cases['error'].max()),
        'max_overestimation': float(extreme_cases['error'].min()),
        'mean_abs_error': float(extreme_cases['abs_error'].mean()),
        'median_abs_error': float(extreme_cases['abs_error'].median()),
        'mean_pct_error': float(extreme_cases['pct_error'].mean()),
        'threshold_temperature': float(threshold),
        'num_extreme_cases': int(len(extreme_cases)),
        'extreme_metrics': extreme_metrics
    }
    
    # Create plots for extreme cases
    if len(extreme_cases) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot actual vs predicted for extreme cases
        ax1.scatter(extreme_cases[actual_column], extreme_cases[pred_column], alpha=0.5)
        min_val = min(extreme_cases[actual_column].min(), extreme_cases[pred_column].min())
        max_val = max(extreme_cases[actual_column].max(), extreme_cases[pred_column].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r-')
        ax1.set_title('Actual vs. Predicted Values (Extreme Cases)')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.grid(True)
        
        # Plot error distribution for extreme cases
        sns.histplot(extreme_cases['error'], kde=True, ax=ax2)
        ax2.set_title('Error Distribution (Extreme Cases)')
        ax2.set_xlabel('Error (Actual - Predicted)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save or display
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Extreme cases plot saved to {output_path}")
        else:
            plt.show()
    
    return error_stats

if __name__ == "__main__":
    # Example usage
    project_dir = Path(__file__).resolve().parents[2]
    
    # Print the detected project directory
    print(f"Project directory: {project_dir}")
    
    # Define paths
    predictions_path = project_dir / "data" / "predictions" / "test_predictions.csv"
    reports_dir = project_dir / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    evaluation_results_path = reports_dir / "model_evaluation.json"
    prediction_plot_path = figures_dir / "predictions_plot.png"
    horizon_metrics_plot_path = figures_dir / "horizon_metrics.png"
    residuals_plot_path = figures_dir / "residuals_plot.png"
    extreme_cases_plot_path = figures_dir / "extreme_cases.png"
    
    # Load predictions
    print(f"Loading predictions from {predictions_path}")
    predictions_df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    
    # Evaluate model
    evaluation_results = evaluate_model(
        predictions_df=predictions_df,
        actual_column='actual',
        pred_column='predicted',
        by_horizon=True,
        by_time='hour'
    )
    
    # Save evaluation results
    save_evaluation_results(evaluation_results, str(evaluation_results_path))
    
    # Plot predictions
    plot_predictions(
        predictions_df=predictions_df,
        actual_column='actual',
        pred_column='predicted',
        sequence_id_column='sequence_id',
        num_sequences=5,
        output_path=str(prediction_plot_path)
    )
    
    # Plot metrics by horizon
    plot_metrics_by_horizon(
        evaluation_results=evaluation_results,
        output_path=str(horizon_metrics_plot_path)
    )
    
    # Plot residuals
    plot_residuals(
        predictions_df=predictions_df,
        actual_column='actual',
        pred_column='predicted',
        output_path=str(residuals_plot_path)
    )
    
    # Analyze extreme cases
    extreme_analysis = analyze_extreme_cases(
        predictions_df=predictions_df,
        actual_column='actual',
        pred_column='predicted',
        percentile_threshold=90,
        output_path=str(extreme_cases_plot_path)
    )
    
    # Save extreme analysis results
    extreme_analysis_path = reports_dir / "extreme_cases_analysis.json"
    with open(extreme_analysis_path, 'w') as f:
        json.dump(extreme_analysis, f, indent=4)
    
    print("Model evaluation completed")