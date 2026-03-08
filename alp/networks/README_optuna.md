# ALP Optuna Hyperparameter Optimization Wrapper

This module provides an easy-to-use interface for Optuna hyperparameter optimization with ALP neural networks. It's designed to minimize the amount of code needed to run effective hyperparameter optimization.

## Quick Start

### 1. Basic Usage (3 lines of code!)

```python
from alp.networks import quick_optimize, MLP
from alp.data.datasets import load_lsst_data, preprocess_lsst_data

def load_data():
    z, mu, err = load_lsst_data()
    return preprocess_lsst_data(z, mu, err, train_split=0.8)

results = quick_optimize(
    model_class=MLP,
    model_kwargs={'n_inputs': 1, 'n_outputs': 2},
    data_loader=load_data,
    n_trials=50
)
```

### 2. Advanced Usage with Custom Search Space

```python
from alp.networks import OptunaOptimizer
import optuna

class CustomOptimizer(OptunaOptimizer):
    def suggest_hyperparameters(self, trial):
        return {
            'deep': trial.suggest_categorical('deep', [[100, 100], [200, 200, 200]]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            'actfn': trial.suggest_categorical('actfn', ['relu', 'tanh'])
        }

optimizer = CustomOptimizer(
    model_class=MLP,
    model_kwargs={'n_inputs': 1, 'n_outputs': 2},
    data_loader=load_data
)

study = optimizer.optimize(
    n_trials=100,
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner()
)
```

## Features

- **Minimal setup**: Optimize with just a few lines of code
- **Customizable**: Easy to override hyperparameter search spaces
- **Smart defaults**: Uses NSGA-II sampling and Median pruning by default
- **Automatic plotting**: Generates optimization visualizations automatically
- **Results saving**: Saves best hyperparameters and study summaries to JSON
- **Error handling**: Robust error handling with automatic trial failure recovery

## API Reference

### `quick_optimize(model_class, model_kwargs, data_loader, n_trials=50, output_dir='outputs')`

**Quick optimization function with minimal setup.**

**Parameters:**
- `model_class`: ALP model class (e.g., MLP)
- `model_kwargs`: Fixed model parameters (n_inputs, n_outputs, etc.)
- `data_loader`: Function returning (X_train, X_val, y_train, y_val)
- `n_trials`: Number of optimization trials (default: 50)
- `output_dir`: Directory to save results (default: 'outputs')

**Returns:**
- `dict`: {'best_params': {...}, 'best_value': float, 'study': optuna.Study}

### `OptunaOptimizer`

**Main optimizer class with full customization options.**

**Methods:**
- `suggest_hyperparameters(trial)`: Override to customize search space
- `create_model(hyperparams)`: Override to customize model creation  
- `optimize(n_trials, timeout, study_name, sampler, pruner, output_dir)`: Run optimization
- `plot_results(study, output_dir)`: Generate optimization plots

## Examples in ALP

- `experiments/lsst/quick_optuna_example.py`: Minimal setup example
- `experiments/lsst/advanced_optuna_example.py`: Custom search space example
- `experiments/lsst/optuna_lsst.py`: Original detailed implementation

## Default Hyperparameter Search Space

By default, the optimizer searches over:
- **Architecture**: Various layer configurations ([100,100,100], [200,200,200,200], etc.)
- **Dropout**: 0.05 to 0.3 
- **Learning rate**: 1e-5 to 1e-2 (log-uniform)
- **Batch size**: [16, 32, 64, 128]

## Output Files

The optimizer automatically saves:
- `best_hyperparameters.json`: Best hyperparameters and performance
- `study_summary.json`: Complete study information
- `optuna_results.png`: Optimization visualization plots

## Customization Tips

1. **Custom Architecture Search**: Override `suggest_hyperparameters()` 
2. **Custom Model Creation**: Override `create_model()` for custom compilation
3. **Advanced Sampling**: Use TPESampler, CmaEsSampler, etc.
4. **Aggressive Pruning**: Use HyperbandPruner for faster optimization
5. **Multi-objective**: Extend for optimizing multiple metrics

## Requirements

- `optuna>=3.0.0`
- `tensorflow>=2.8.0`
- `numpy`
- `matplotlib` (for plotting)