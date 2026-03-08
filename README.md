# Astro Layer Perceptron (ALP)

A modular deep learning framework for astrophysical data modeling and
cosmological parameter estimation.

ALP provides neural-network tools for reconstructing cosmological
observables and performing data-driven inference with uncertainty
quantification.

------------------------------------------------------------------------

# Overview

**ALP (Astro Layer Perceptron)** is a Python framework designed for
regression problems in astrophysics and cosmology. It implements neural
network architectures optimized for reconstructing cosmological
functions from observational data.

The framework focuses on:

-   cosmological data reconstruction
-   uncertainty quantification
-   neural-network--based inference
-   hyperparameter optimization with evolutionary algorithms

ALP consolidates methodologies previously developed in several research
works into a reusable, modular software library.

------------------------------------------------------------------------

# Features

-   **MLP architectures** for astrophysical regression tasks
-   **Monte Carlo Dropout (MCDO)** for uncertainty quantification
-   **Genetic Algorithm (GA) hyperparameter optimization**
-   **Physics utilities** for cosmological calculations
-   **Data preprocessing pipelines** for astrophysical datasets
-   Modular structure designed for experimentation and reproducibility

------------------------------------------------------------------------

# Quick Start

``` python
import tensorflow as tf
from alp.networks.mlp import MLP
from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.utils.gpu_config import setup_tensorflow_for_training

# Use CPU to avoid GPU issues
setup_tensorflow_for_training(force_cpu=True)

# Load data
z_data, mu_data, error_data = load_lsst_data()

z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(
    z_data,
    mu_data,
    error_data
)

# Create model
model = MLP(
    n_inputs=1,
    deep=[200, 200, 200, 200],
    dropout=0.1,
    mcdropout=True,
    n_outputs=2
)

keras_model = model.model_tf()

# Train
keras_model.compile(optimizer="adam", loss="mse")
keras_model.fit(z_train, y_train, validation_data=(z_test, y_test))
```

------------------------------------------------------------------------

# Running Experiments

Example scripts for common astrophysical analyses are available in the
`experiments` directory.

### LSST Supernova analysis

``` bash
python experiments/lsst/train_lsst.py
```

### Cosmic Chronometers (H(z))

``` bash
python experiments/cc/training.py
```

### Hyperparameter optimization

``` bash
python experiments/lsst/optuna_lsst.py
```

------------------------------------------------------------------------

# Project Structure

    ALP/
    ├── alp/                    # Main package
    │   ├── physics/            # Cosmology calculations
    │   ├── data/               # Data loading & preprocessing
    │   ├── networks/           # Neural network architectures
    │   └── utils/              # Utilities
    │
    ├── experiments/            # Example experiments
    │   ├── lsst/               # Supernova Ia analysis
    │   └── cc/                 # Cosmic chronometers
    │
    └── data/                   # Sample datasets

------------------------------------------------------------------------

# Installation

Create a dedicated environment:

``` bash
conda create -n alp python=3.9
conda activate alp
```

Install the package:

``` bash
pip install -e .
```

------------------------------------------------------------------------

# Scientific Background

The methodologies implemented in ALP were developed through a series of
works focused on neural-network reconstructions of cosmological
observables.

The initial framework introduced neural networks with Monte Carlo
Dropout uncertainty quantification for reconstructing cosmological
quantities such as the Hubble parameter, growth rate, and supernova
distance modulus:

- Gómez-Vargas, I., Vázquez, J. A., Esquivel, R. M., & García-Salcedo, R.
(2023).\
Neural Network Reconstructions for the Hubble Parameter, Growth Rate and
Distance Modulus.\
European Physical Journal C 83, 304.

This approach was later extended by introducing Genetic Algorithms (GA)
to optimize neural network architectures and hyperparameters for
cosmological regression tasks:

- Gómez-Vargas, I., Andrade, J. B., & Vázquez, J. A. (2023).\
Neural Networks Optimized by Genetic Algorithms in Cosmology.\
Physical Review D 107, 043509.

The combined GA + MCDO framework has been applied in several
astrophysical contexts, including:

- Garcia-Arroyo, G., Gómez-Vargas, I., & Vázquez, J. A. (2026).\
Data-driven modeling of rotation curves with artificial neural
networks.\
Physics of the Dark Universe.

- Mitra, A., Gómez-Vargas, I., & Zarikas, V. (2024).\
Dark energy reconstruction analysis with artificial neural networks:
Application on simulated Supernova Ia data from Rubin Observatory.\
Physics of the Dark Universe.

- Gómez-Vargas, I., & Vázquez, J. A. (2024).\
Deep Learning and genetic algorithms for cosmological Bayesian inference
speed-up.\
Physical Review D 110, 083518.

ALP consolidates these methodologies into a modular Python library,
replacing the notebook-based implementations used in the original
studies and enabling reproducible research and future extensions.

------------------------------------------------------------------------

# Citation

If you use ALP in scientific work, please cite the methodological
papers:

```bibtex
@article{GomezVargas2023,
  title={Neural networks optimized by genetic algorithms in cosmology},
  author={Gómez-Vargas, I. and Andrade, J. B. and Vázquez, J. A.},
  journal={Physical Review D},
  volume={107},
  number={4},
  pages={043509},
  year={2023},
  publisher={American Physical Society},
  doi={https://doi.org/10.1103/PhysRevD.107.043509},
  url={https://doi.org/10.48550/arXiv.2209.02685}
}
```

```bibtex

@article{gomez2023neural,
  title={Neural network reconstructions for the Hubble parameter, growth rate and distance modulus},
  author={G{\'o}mez-Vargas, Isidro and Medel-Esquivel, Ricardo and Garc{\'\i}a-Salcedo, Ricardo and V{\'a}zquez, J Alberto},
  journal={The European Physical Journal C},
  volume={83},
  number={4},
  pages={304},
  year={2023},
  publisher={Springer}
}
```

A dedicated software citation will be added in future releases.

------------------------------------------------------------------------

# License

This project is released under the MIT License.
