# Oil Temperature Forecasting Project

[![Documentation](https://img.shields.io/badge/Documentation-Full_Docs-blue)](./Documentation.md)

A machine learning system for predicting transformer oil temperature (OT) 24 hours in advance to help optimize energy distribution and prevent equipment damage.

## Project Overview

Electrical transformers are critical infrastructure components that require careful monitoring. Oil temperature is a key indicator of transformer health, with extreme temperatures potentially leading to equipment failure. This project implements a deep learning approach to forecast oil temperature at hourly intervals for the next 24 hours.

**Key Features:**

- Sophisticated preprocessing pipeline for handling missing values and outliers
- Extensive feature engineering including temporal, load-based, and Fourier features
- Encoder-decoder LSTM architecture with attention mechanism
- Special handling for extreme temperature events
- Comprehensive evaluation framework

For detailed technical information, please see the [full documentation](./documentation.md).

## Folders Structure

```
OilTemperatureForecast
├── Env                  - Python virtual environment
├── Data
│   ├── raw              - Original input data
│   ├── preprocessed     - Cleaned data
│   └── features         - Feature-engineered datasets
├── models               - Trained model files
│   ├── report           - Model evaluation reports
├── notebooks            - Jupyter notebooks for EDA
├── src
│   ├── data             - Data preprocessing pipeline
│   ├── features         - Feature engineering code
│   ├── models           - Model training and evaluation
│   └── visualization    - Results visualization
├── requirements.txt     - Project dependencies
├── README.md            - This file
├── documentation.md     - Detailed technical documentation
├── .gitignore           - Git ignore file
└── setup.py             - Installation script
```

## Pre-requisites

This project requires **Python 3.12.10**. A GPU is **not required** for running the models, though it may speed up training time. We strongly recommend using a virtual environment to avoid dependency conflicts.

### Setting up the environment

**Windows:**

```bash
python -m venv Env
.\Env\Scripts\activate
```

**Linux/Mac:**

```bash
python -m venv Env
source Env/bin/activate
```

## Installation & Usage

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Explore the Data

The exploratory data analysis can be found in the Jupyter notebook:

```
./notebooks/data_exploration.ipynb
```

### 3. Run the Pipeline

The full pipeline can be executed step by step:

#### Preprocess Data

```bash
python -m src.data.make_dataset
```

#### Extract Features

```bash
python -m src.features.build_features
```

#### Train Models

```bash
python -m src.models.train_model
```

#### Evaluate Models

```bash
python -m src.models.evaluate_model
```

#### Make Predictions

```bash
python -m src.models.predict
```

#### Visualize Results

```bash
python -m src.visualization.visualize
```

### 4. Alternative: Run All Steps

Instead of running each step separately, you can install the package and run the entire pipeline:

```bash
# Basic installation
pip install -e .

# Alternative installation methods
pip install .

# Install with development dependencies
pip install '.[dev]'

# Create distribution packages
python setup.py sdist bdist_wheel

# Installation from source
python setup.py install

# Using build package
pip install build
python -m build
```

## Current Status

This implementation is currently a proof-of-concept and requires further optimization before production use. The system shows promising performance for normal operating conditions but needs improvement for extreme temperature prediction. See the documentation for detailed performance metrics and planned enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to the team that provided the transformer data
- This project was developed as part of the Time Series Forecasting Challenge
