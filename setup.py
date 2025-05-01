from setuptools import setup, find_packages

setup(
    name="OilTemperatureForecast",
    version="0.1.0",
    description="Forecasting transformer oil temperature using machine learning",
    author="",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.12.0",
    install_requires=[
        "pandas==2.2.3",
        "numpy==2.1.3",
        "scikit-learn==1.6.1",
        "matplotlib==3.10.1",
        "seaborn==0.13.2",
        "scipy==1.15.2",
        "tensorflow==2.19.0",
        "plotly==6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "make-dataset=src.data.make_dataset:main",
            "build-features=src.features.build_features:engineer_features",
            "train-model=src.models.train_model:main",
            "evaluate-model=src.models.evaluate_model:main",
            "predict=src.models.predict_model:main",
            "visualize=src.visualization.visualize:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)