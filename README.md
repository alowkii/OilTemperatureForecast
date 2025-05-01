## Folders structure

```
OilTempearatureForecast
├── Env
├── Data
│   ├── raw
│   ├── preprocessed
│   └── features
├── models
├── notebooks
├── src
│   ├── data
│   ├── features
│   ├── models
│   ├── visualization
│   └── utils
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py
```

## Pre-requisites

This project is designed to be run in a Python 3.12.10 environment. It is recommended to use a virtual environment to avoid conflicts with other projects.

Windows:

```bash
python -m venv Env
.\Env\Scripts\activate
```

or on Linux/Mac:

```bash
python -m venv Env
source venv/bin/activate
```

## 1. Install requirements

Python version: Python 3.12.10

### Run:

```bash
pip install -r requirements.txt
```

## 2. EDA at:

./notebooks/data_exploration.ipynb

## 3. Preprocess files:

### Run:

```bash
python -m src.data.make_dataset
```

## 4. Extract features:

### Run:

```bash
python -m src.features.build_features
```

## 5. Train models:

### Run:

```bash
python -m src.models.train_model
```

## 6. Evaluate models:

### Run:

```bash
python -m src.models.evaluate_model
```

## 7. Make predictions on test set:

```bash
python -m src.models.predict
```

## 8. Visualize results:

### Run:

```bash
python -m src.visualization.visualize
```

## 9. Run all steps in order:

### Run:

```bash
pip install -e .
pip install .
pip install '.[dev]' # to install development dependencies
python setup.py sdist bdist_wheel # to create a source distribution and a wheel distribution
python setup.py install # to install the package

# for build
pip install build
python -m build
```
