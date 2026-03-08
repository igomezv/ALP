# ALP - Astro Layer Perceptron

A deep learning framework for astrophysical data modeling and cosmological parameter estimation.

## Overview

ALP (Astro Layer Perceptron) is a modular Python framework for astrophysical regression tasks, particularly supernova data analysis and cosmological parameter estimation using neural networks with uncertainty quantification.

## Quick Start

```python
import tensorflow as tf
from alp.networks.mlp import MLP
from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.utils.gpu_config import setup_tensorflow_for_training

# Use CPU to avoid GPU issues
setup_tensorflow_for_training(force_cpu=True)

# Load data
z_data, mu_data, error_data = load_lsst_data()
z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(z_data, mu_data, error_data)

# Create and train model
model = MLP(n_inputs=1, deep=[200, 200, 200, 200], dropout=0.1, mcdropout=True, n_outputs=2)
keras_model = model.model_tf()
keras_model.compile(optimizer='adam', loss='mse')
keras_model.fit(z_train, y_train, validation_data=(z_test, y_test))
```

## Running Experiments

```bash
# LSST supernova analysis
python experiments/lsst/train_lsst.py

# Cosmic chronometers (H(z))
python experiments/cc/training.py

# Hyperparameter optimization
python experiments/lsst/optuna_lsst.py
```

## Project Structure

```
ALP/
├── alp/                    # Main package
│   ├── physics/           # Cosmology calculations
│   ├── data/              # Data loading & preprocessing
│   ├── networks/          # Neural network architectures
│   └── utils/             # Utilities
├── experiments/           # Experiments
│   ├── lsst/              # Supernova analysis
│   └── cc/                # Cosmic chronometers
└── data/                  # Sample datasets
```

## Installation

```bash
conda create -n alp python=3.9
conda activate alp
pip install -e .
```

## Features

- **MLP Networks**: Multi-layer perceptrons with MC Dropout
- **Uncertainty Quantification**: MC Dropout, heteroscedastic loss, conformal prediction
- **Physics Module**: ΛCDM calculations, distance modulus
- **Data Utilities**: Train/test splits with endpoint preservation
