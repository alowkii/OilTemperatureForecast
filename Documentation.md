# Oil Temperature Forecasting Documentation

## Overview

This project aim to solve the Time Series Challenge where the objective was forecasting oil temperature (OT) of electricity transformers hourly for the next 24 hours. The accurate prediction of oil temperature is essential for transformer management, as it helps preventing equipment damage and optimising energy distribution across the grid.

## Project Structure

The project follows a standard structure with seperate directories for data, models, and source code:

```
OilTemperatureForecast/
├── Data/
│   ├── features/        - Contains feature-engineered datasets
│   ├── predictions/     - Stores model prediction results
│   ├── preprocessed/    - Has cleaned and preprocessed data
│   └── raw/             - Original input data files
├── models/              - Trained models and there artifacts
│   ├── report/          - Reports specific to models
│   ├── lstm_encoder_decoder_model_history.json
│   ├── lstm_encoder_decoder_model.keras
│   ├── lstm_model_history.json
│   ├── robust_scaler.pkl
│   └── scaler.pkl
├── notebooks/
│   └── data_exploration.ipynb - For exploration data analysis
├── reports/             - Results and visualisations
│   ├── figures/         - Generated image outputs
│   ├── dashboard.html   - Simple web dashboard
│   ├── extreme_cases_analysis.json
│   └── model_evaluation.json
├── src/                 - All source code
│   ├── data/            - Data loading/preprocessing functions
│   ├── features/        - Feature engineering code
│   ├── models/          - Model training and evaluation
│   └── visualization/   - Code for creating visualisations
├── .gitignore           - Specifies ignored files
├── README.md            - Basic project documentation
├── requirements.txt     - Required packages
└── setup.py             - Project installation script
```

## Requirements

- Python 3.12.10
- These packages are required:
  - pandas==2.2.3
  - numpy==2.1.3
  - scikit-learn==1.6.1
  - matplotlib==3.10.1
  - seaborn==0.13.2
  - scipy==1.15.2
  - tensorflow==2.19.0
  - plotly==6.0.1

## 1. Data Preprocessing

### Approaches Used

The preprocessing pipeline implemented in `src/data/make_dataset.py` uses several stratergies to prepare data for modeling:

#### Missing Value Handling

A multi-stage approach was developed for addressing missing values:

1. **Forward Fill for Small Gaps**: First, the missing values get filled using last known value for gaps up to 4 time steps
2. **Time-based Interpolation**: For medium gaps (< max_gap_limit), the code uses temporal interpolation
3. **Isolation of Large Gaps**: For very large gaps (≥ max_gap_limit), no interpolation is attempted to avoid introducing misleading patterns in data

```python
# For small gaps - forward fill with limit
processed_df = processed_df.ffill(limit=4)

# For medium gaps - use time interpolation with limits
mask = processed_df[col].isnull()
runs = mask.ne(mask.shift()).cumsum()
run_sizes = mask.groupby(runs).sum()
large_gaps = run_sizes[run_sizes > max_gap_limit].index
small_gap_mask = mask & ~runs.isin(large_gaps)
processed_df.loc[small_gap_mask, col] = temp_series.loc[small_gap_mask]
```

#### Outlier Detection and Handling

The developer implemented IQR-based methods with different thresholds for different variables:

1. **Oil Temperature (OT)**:

   - More permissive thresholds: Q1 - 3.0 _ IQR to Q3 + 3.0 _ IQR
   - Values outside these bounds are capped rather than replaced to keep extreme patterns

2. **Other Variables**:
   - Standard thresholds: Q1 - 2.5 _ IQR to Q3 + 2.5 _ IQR
   - Outliers get replaced with the column median

```python
# Special handling for oil temperature
lower_bound = Q1 - 3.0 * IQR
upper_bound = Q3 + 3.0 * IQR
processed_df.loc[processed_df[column] < lower_bound, column] = lower_bound
processed_df.loc[processed_df[column] > upper_bound, column] = upper_bound
```

#### Justification of Preprocessing Choices

- **Different Ways to Handle Missing Values**: Time series data often has temporal patterns that helps with imputation. The tiered approach respects these patterns while avoiding adding artifacts from imputing over massive gaps.

- **Special Treatment for Oil Temperature**: Since OT is the target, its important to preserve extreme patterns for forecasting high-risk scenarios. Capping rather than replacing outliers lets the model learn from these extreme but valid data points.

- **Time Feature Creation**: Converting datetime to cyclical features keeps the continuity of time variables, so the model can capture patterns related to time better.

## 2. Feature Engineering

The feature engineering module (`src/features/build_features.py`) creates many types of features to improve the model's ability for capturing temporal patterns, load relationships, and extreme events.

### Key Feature Types

#### Temporal Dependencies

```python
# Creating lag features to capture autocorrelation
for col in columns:
    for lag in lag_periods:
        lag_name = f"{col}_lag_{lag}"
        df_with_lags[lag_name] = df_with_lags[col].shift(lag)
```

**Rationale**: Oil temperature shows strong autocorrelation patterns. The lag features for 1, 2, 3, 6, 12, 24, and 48 hours helps the model learn from recent history and seeing daily patterns.

#### Rolling Window Statistics

```python
# Statistical values over different time windows
for window in windows:
    rolling = df_with_rolling[col].rolling(window=window, min_periods=1)

    for func_name, func in functions.items():
        feature_name = f"{col}_rolling_{window}_{func_name}"
        df_with_rolling[feature_name] = rolling.apply(func, raw=True)
```

**Rationale**: Rolling statistics (mean, std, min, max) give information about recent trends, volatility, and extremes, that helps the model understand current value context.

#### Seasonality Modeling

```python
# Fourier features for capturing multiple seasonal patterns
for k in range(1, order + 1):
    df_with_fourier[f'{period_name}_sin_{k}'] = np.sin(2 * k * np.pi * hour_var / period)
    df_with_fourier[f'{period_name}_cos_{k}'] = np.cos(2 * k * np.pi * hour_var / period)
```

**Rationale**: Transformers have daily (24-hour), weekly (168-hour), and yearly (8760-hour) patterns. These Fourier features efficiently capture cyclical behaviors without needing years of data.

#### Load Relationship Features

```python
# Creating ratios between useful and useless loads
df_with_ratios['load_efficiency'] = (
    df_with_ratios['total_useful_load'] / df_with_ratios['total_useless_load']
)
```

**Rationale**: The ratio between useful and useless loads acts as key indicator of transformer efficiency. These features help model understand how different load distributions affect temperature.

#### Temperature Rate Features

```python
# Rate of change in temperature over time
df_with_rates[f'{temperature_column}_roc_{window}'] = df[temperature_column].diff(window) / window
```

**Rationale**: The rate of temperature change is critical for predicting fast heating or cooling events. These features captures acceleration and deceleration patterns in oil temperature.

#### Extreme Temperature Features

```python
# Binary indicator of extreme temperatures
new_features[f'{temperature_column}_extreme'] = (df[temperature_column] >= threshold).astype(int)

# Tracking time since last extreme temperature event
hours_since_extreme = np.full(len(df), len(df))
for i, current_time in enumerate(df.index):
    past_extremes = extreme_events[extreme_events < current_time]
    if len(past_extremes) > 0:
        most_recent = past_extremes[-1]
        hours_diff = (current_time - most_recent).total_seconds() / 3600
        hours_since_extreme[i] = hours_diff
```

**Rationale**: Extreme temperature events are rare but critical for transformer management. These features target prediction of high-temperature scenarios by tracking patterns before extreme events.

### Justification of Feature Engineering Approach

The feature engineering approach was designed for addressing multiple aspects of the oil temperature forecasting problem:

1. **Temporal Context**: Lag and rolling features gives model information about recent history and trends.

2. **Periodicity**: Fourier features capture daily, weekly, yearly patterns without needing excessive historical data.

3. **Load Relationships**: Load ratio and difference features helps model understand how transformer load affects oil temperature.

4. **Extreme Events**: Special features for extreme temperatures helps model better predict high-risk scenarios.

5. **Rate of Change**: Temperature rate features enables model to capture acceleration and deceleration patterns.

## 3. Data Exploration and Insights

Key findings from exploratory data analysis done in `notebooks/data_exploration.ipynb`:

### Temporal Patterns in Oil Temperature

The analysis revealed strong patterns in oil temperature:

- **Daily Cycle**: Highest temperatures happen in afternoon (2-5 PM), with lowest temperatures early morning (4-6 AM).
- **Weekly Pattern**: Weekdays has higher average temperatures than weekends, probably due to more industrial activity.
- **Seasonal Effects**: Summer months shows higher average temperatures and more extreme events than winter months.

### Load-Temperature Relationship

Analysis of relationship between different load types and oil temperature showed:

- **High Correlation with Useful Loads**: HUFL (High Useful Load) had strongest correlation with oil temperature (r = 0.78), followed by MUFL (Middle Useful Load, r = 0.65).
- **Weaker Correlation with Useless Loads**: Useless loads (HULL, MULL, LULL) showed weaker correlations (r < 0.5), suggesting less direct impact on heating.
- **Load Efficiency Impact**: The ratio between useful and useless loads was strong predictor of temperature changes, with higher ratios associated with faster temperature increases.

### Extreme Temperature Events

Analysis of extreme temperature events (defined as temperatures above 95th percentile):

- **Duration**: Most extreme events lasted 3-5 hours, with few extending to 8+ hours.
- **Precursors**: Extreme events typically had rapid increase in HUFL/HULL ratio in the previous 6 hours.
- **Time Distribution**: 78% of extreme events occured on weekdays, with 65% happening between 1 PM and 6 PM.

### Autocorrelation Analysis

Strong autocorrelation was observed in oil temperature time series:

- **Short-term**: Very high autocorrelation (> 0.9) for lags up to 12 hours
- **Daily Pattern**: Clear daily cycle with peaks at 24, 48, and 72-hour lags
- **Gradual Decay**: Autocorrelation remained significant (> 0.3) even at 7-day lags

These insights directly informed the feature engineering approach, especially the lag features, load ratio features, and special handling of extreme temperature events.

## 4. Model Building

### Architecture Overview

The project implements an encoder-decoder LSTM architecture with attention mechanism, designed specifically for multi-step forecasting task (24-hour prediction horizon).

#### Encoder Component

```python
# Bidirectional LSTM layers for encoder
lstm_layer = Bidirectional(
    LSTM(
        units,
        return_sequences=return_sequences,
        return_state=True,
        recurrent_dropout=recurrent_dropout,
        name=f'encoder_lstm_{i}'
    )
)
```

The encoder processes input sequence (24 hours of historical data) using bidirectional LSTM layers to capture patterns from both past and future context.

#### Attention Mechanism

```python
# Apply attention between encoder and decoder
attention_layer = Attention(name='attention_layer')
context_vector = attention_layer([decoder_outputs, encoder])
```

The attention mechanism allows model to focus on most relevant parts of input sequence when making predictions, particularly useful for identifying patterns that come before temperature spikes.

#### Decoder Component

```python
# Decoder using LSTM and dense layers
decoder_lstm = LSTM(
    decoder_units[0] * 2,  # Match bidirectional encoder state size
    return_sequences=True,
    return_state=True,
    recurrent_dropout=recurrent_dropout,
    name='decoder_lstm'
)
```

The decoder generates output sequence (24-hour forecast) using LSTM layers and dense connections, with context vector from attention mechanism providing additional information.

#### Custom Loss Function

```python
def temporal_weighted_mse():
    def loss_fn(y_true, y_pred):
        # Calculate squared error first
        squared_error = tf.square(y_true - y_pred)

        # Setting weights for different time steps in prediction horizon
        # Higher weights for early predictions and day boundaries
        weights = tf.ones_like(y_true)

        # Emphasizing first 6 steps (critical short-term forecast)
        weights = tf.tensor_scatter_nd_update(
            weights,
            indices=tf.constant([[i, j] for i in range(tf.shape(y_true)[0]) for j in range(6)]),
            updates=tf.constant([1.5] * (tf.shape(y_true)[0] * 6))
        )
```

A custom loss function was implemented to place higher importance on:

1. Near-term predictions (first 6 hours), which are more actionable for operators
2. Day boundary predictions (hours 23-24), which are critical for planning next day activities

### Model Training

The model training process included:

- Sequence creation with balanced representation of extreme values
- Early stopping and learning rate reduction to prevent overfitting
- Horizon weighting to prioritize accuracy of earlier predictions
- Encoder-decoder with attention for capturing complex patterns

## 5. Model Evaluation

### Performance Metrics

The model was evaluated using multiple metrics to understand its performance across different conditions:

#### Overall Performance

From `reports/model_evaluation.json`:

| Metric | Value  |
| ------ | ------ |
| MAE    | 13.27  |
| RMSE   | 14.67  |
| MAPE   | 34.25% |
| R²     | -4.43  |

#### Performance by Forecast Horizon

The model shows increasing error with longer forecast horizons:

| Horizon (hours)    | MAE   | RMSE  | MAPE   |
| ------------------ | ----- | ----- | ------ |
| 0 (1-hour ahead)   | 12.98 | 14.16 | 33.93% |
| 6                  | 13.25 | 14.59 | 34.28% |
| 12                 | 13.48 | 14.89 | 34.68% |
| 18                 | 13.28 | 14.79 | 34.14% |
| 23 (24-hour ahead) | 13.45 | 15.06 | 34.44% |

#### Performance by Time of Day

Error rates vary significantly depending on hour of day:

| Hour  | MAE   | RMSE  | MAPE   |
| ----- | ----- | ----- | ------ |
| 4 AM  | 10.20 | 10.94 | 29.42% |
| 12 PM | 15.91 | 17.06 | 38.70% |
| 3 PM  | 17.84 | 19.31 | 40.93% |
| 8 PM  | 15.86 | 17.35 | 37.91% |

#### Extreme Temperature Performance

From `reports/extreme_cases_analysis.json`:

| Metric         | Value   |
| -------------- | ------- |
| Mean Error     | 24.11   |
| Mean Abs Error | 24.11   |
| Mean Pct Error | 49.83%  |
| RMSE           | 24.28   |
| R²             | -328.66 |

### Analysis of Model Strengths and Weaknesses

#### Strengths:

1. **Consistent Performance Across Horizons**: The model maintains relatively stable MAE and RMSE across different forecast horizons, with only moderate degreedation at longer horizons.

2. **Day/Night Pattern Recognition**: The model performs better during nighttime hours (10PM-6AM) when temperatures are more stable, showing it has learning the daily cycle.

3. **Attention Mechanism Benefit**: The attention mechanism helps model focus on relevant patterns, particularly useful for capturing the relationship between load changes and temperature responses.

#### Weaknesses:

1. **Poor Extreme Temperature Prediction**: The model significantly underestimates extreme temperatures, with mean error of 24.11°C for extreme cases and terrible R² of -328.66.

2. **Negative Overall R²**: The overall R² of -4.43 indicates model performs worse than a naive baseline (such as predicting mean temperature).

3. **High Error During Peak Hours**: Performance is worst during afternoon hours (1-5 PM) when temperatures typically peak, precisely when accurate forecasts is most critical.

4. **High MAPE**: The MAPE of 34.25% indicates large percentage errors, making forecasts less reliable for operational planning.

### Potential Improvements

1. **Model Architecture Refinements**:

   - Implement ensemble methods combining predictions from specialized models
   - Develop dedicated model for extreme temperature events
   - Consider simpler models alongside complex ones to improve baseline performance

2. **Loss Function Enhancements**:

   - Implement asymmetric loss functions that penalize under-prediction of high temperatures more severely
   - Develop quantile regression for better uncertainty estimation

3. **Feature Selection and Engineering**:

   - Perform feature importance analysis to identify and focus on most predictive features
   - Engineer additional features specifically targeting extreme temperature patterns

4. **Data Augmentation**:

   - Generate synthetic samples for extreme temperature scenarios
   - Apply time series specific augmentation techniques to increase prevalence of extreme events in training

5. **Time-based Model Selection**:
   - Implement different models for different times of day or temperature ranges
   - Create meta-model that selects appropriate forecasting model based on recent conditions

## 6. Setup and Usage Instructions

### Environment Setup

1. Create Python 3.12.10 virtual environment:

   ```bash
   python -m venv Env_py312
   source Env_py312/bin/activate  # Linux/Mac
   .\Env_py312\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

1. Preprocess data:

   ```bash
   python -m src.data.make_dataset
   ```

2. Extract features:

   ```bash
   python -m src.features.build_features
   ```

3. Train model:

   ```bash
   python -m src.models.train_model
   ```

4. Evaluate model:

   ```bash
   python -m src.models.evaluate_model
   ```

5. Make predictions:

   ```bash
   python -m src.models.predict
   ```

6. Visualize results:
   ```bash
   python -m src.visualization.visualize
   ```

Alternatively, install package to run all components:

```bash
pip install -e .
```

## Assumptions

1. The data follows consistent sampling frequency (hourly readings).
2. The relationship between load variables and oil temperature remain relatively stable over time.
3. Extreme temperatures follow similar patterns to normal temperatures but with different magnitude.
4. The 24-hour forecast horizon is sufficient for operational planning and maintenance scheduling.
5. Missing values in data are not systematic but random or related to known maintenance periods.

## Ongoing Work and Optimization Needs

It is important to emphasise that this system is still in early development stages and requires significant optimization. The current implementation should be considered a proof-of-concept rather than a production-ready solution. Several areas need substantial refinement:

1. **Hyperparameter Tuning**: The model hyperparameters have not been optimised to their finest settings. Initial experiments were conducted with limited hyperparameter exploration, and a comprehensive grid search or Bayesian optimization approach is needed.

2. **Model Architecture Experimentation**: While the encoder-decoder architecture was implemented, many variations remain unexplored. The number of layers, units per layer, and attention mechanism details all need systematic tuning.

3. **Feature Selection Pipeline**: The current approach includes a large number of features without rigorous selection. Many features may be redundant or even harmful to model performance, requiring proper feature selection techniques.

4. **Computational Efficiency**: The current implementation prioritises functionality over efficiency. The code needs optimization for both training and inference speed, particularly for potential real-time applications.

5. **Scaling Issues**: The model has not been tested with very large datasets, and the current preprocessing pipeline may face challenges with scaling to production volumes.

6. **Extreme Case Handling**: Despite efforts to address extreme temperature events, this remains the most significant limitation requiring dedicated attention and specialized modeling approaches.

## Conclusion

This project implements comprehensive approach to forecasting transformer oil temperature with focus on extreme value prediction. While current model shows limitations in performance, especially for extreme temperatures, the thorough evaluation framework provide valuable insights for iterative improvement.

The most critical area for improvement is the prediction of extreme temperatures, which are particularly important for transformer management and preventing equipment damage. Future work should focus on specialized models or ensemble approaches to better capture these high-risk scenarios.

It should be stressed that this implementation represents only the initial phase of development. The system requires extensive optimization, fine-tuning, and testing before it can be considered ready for real-world deployment. The current results should be interpreted as baseline performance that establishes the foundation for future improvements rather than the final capabilities of the system.
