#### 1. Install requirements

Python version: Python 3.13.1

## Run:

```bash
pip install -r requirements.txt
```

#### 2. EDA at:

./notebooks/data_exploration.ipynb

Based on the exploratory data analysis, here are key findings about the transformer oil temperature dataset:

1. **Data Structure**:

   - The training set consists of approximately 69,696 rows with 15-minute intervals
   - The time span covers data from July 2016 to June 2018
   - The dataset is dense with few to no missing values
   - Data points are consistently spaced at 15-minute intervals

2. **Statistical Properties**:

   - The oil temperature (OT) ranges from near 0°C to around 40°C with a mean of approximately 13°C
   - Approximately 1-2% of data points can be classified as outliers using standard z-score methods
   - Load variables show varying distributions with both positive and negative values
   - Oil temperature distribution shows seasonal patterns with clear upper and lower boundaries

3. **Correlations**:

   - The strongest predictors of oil temperature are MUFL (Middle Useful Load) and HUFL (High Useful Load)
   - Strong autocorrelation exists in the target variable (OT), suggesting good predictability with time-lagged features
   - High multicollinearity exists between the useful load variables (HUFL, MUFL, LUFL)
   - Load ratios (useful vs. useless) might serve as important features

4. **Temporal Patterns**:

   - Clear daily cycles exist with temperatures peaking during midday and dropping at night
   - Weekly patterns show slightly higher temperatures on weekdays than weekends
   - Strong seasonal patterns with higher temperatures in summer months
   - The temporal patterns suggest that time-based features will be valuable for prediction

5. **Preprocessing Recommendations**:

   - Handle outliers using statistical methods (z-score or IQR)
   - Create lag features to capture the strong autocorrelation
   - Add time-based features (hour, day, month) with cyclical encoding
   - Generate rolling statistics to capture recent trends
   - Normalize or standardize features due to their different scales
   - Consider feature interactions, especially between related load variables

6. **Modeling Strategy**:

   - Time series forecasting approaches will be most appropriate
   - LSTM or GRU networks should work well due to the temporal dependencies
   - Multi-step forecasting may be possible given the strong patterns
   - Feature importance analysis should be performed to select most relevant features
   - The strong autocorrelation suggests that even simple models like ARIMA might perform reasonably well as baselines

7. **Key Insights for Feature Engineering**:
   - The ratio between useful and useless loads appears significant
   - Changes in load values correlate with changes in temperature
   - Seasonal decomposition might help isolate underlying patterns
   - Historical values of oil temperature (lagged features) will be crucial predictors
   - Time of day and day of week encode important cyclical patterns

#### 3. Preprocess files:

## Run:

```bash
python -m src.data.make_dataset
```
