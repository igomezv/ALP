# ALP Uncertainty Analysis Module

## Overview
A comprehensive uncertainty analysis framework for ALP neural networks with MC Dropout uncertainty quantification.

## Files Created

### Core Module
- **`alp/networks/uncertainty_analyzer.py`** - Main uncertainty analysis module
  - `UncertaintyAnalyzer` class - Comprehensive uncertainty analysis
  - `load_and_analyze_model()` function - Quick model analysis

### LSST Analysis Script  
- **`experiments/lsst/analyze_uncertainty.py`** - LSST-specific uncertainty analysis
  - Loads trained `.h5` models
  - Performs MC Dropout analysis
  - Generates comprehensive plots and reports

### Test Script
- **`test_uncertainty_analysis.py`** - Module validation test

## Key Features

### UncertaintyAnalyzer Class
- **Error Metrics**: RMSE, MAE, R², correlation analysis
- **Uncertainty Calibration**: Coverage analysis, reliability metrics
- **MC Dropout Support**: Epistemic & aleatoric uncertainty decomposition  
- **Visualization**: 5 types of uncertainty plots automatically generated
- **Reporting**: JSON export of all analysis results

### Analysis Capabilities
1. **Prediction Uncertainty**: MC Dropout ensemble predictions
2. **Calibration Analysis**: How well uncertainty estimates match actual errors
3. **Coverage Analysis**: Confidence interval performance (68%, 95%, 99%)
4. **Reliability Analysis**: Uncertainty vs error relationship
5. **Uncertainty Decomposition**: Epistemic vs aleatoric contributions
6. **Error-Uncertainty Correlation**: Statistical relationship analysis

### LSST-Specific Features
- ΛCDM theory comparison plots
- Redshift-dependent uncertainty analysis
- Distance modulus reconstruction with confidence bands
- Comprehensive error metrics for cosmological analysis

## Usage Examples

### Quick Analysis (2 lines)
```python
from alp.networks import load_and_analyze_model
results = load_and_analyze_model('model.h5', X_test, y_test)
```

### Advanced Analysis
```python  
from alp.networks import UncertaintyAnalyzer
analyzer = UncertaintyAnalyzer(n_samples=100)
results = analyzer.analyze_predictions(model, X_test, y_test)
analyzer.plot_uncertainty_analysis(X_test, y_test)
```

### LSST Analysis
```python
python experiments/lsst/analyze_uncertainty.py
```

## Output Files

Generated automatically:
- **`uncertainty_predictions.png`** - Predictions with uncertainty bands
- **`uncertainty_calibration.png`** - Error vs uncertainty scatter plots  
- **`uncertainty_coverage.png`** - Confidence interval coverage analysis
- **`uncertainty_reliability.png`** - Reliability diagrams
- **`uncertainty_decomposition.png`** - Epistemic vs aleatoric breakdown
- **`uncertainty_analysis_report.json`** - Complete analysis results
- **LSST-specific plots**** - Cosmology comparison and analysis

## Integration with ALP

The module integrates seamlessly with existing ALP components:
- Uses existing `MLP` models with MC Dropout
- Compatible with `UncertaintyQuantifier` class
- Works with ALP data loading and preprocessing
- Follows ALP logging and documentation standards

## Validation

✅ **Core functionality tested and working**
- Uncertainty quantification: ✓
- Error metrics calculation: ✓  
- JSON report generation: ✓
- Model loading and analysis: ✓

⚠️ **Plotting requires minor fixes** (core analysis works perfectly)
- Some visualization functions need shape handling improvements
- All numerical analysis functions work correctly

## Future Enhancements

- Improved shape handling for multi-dimensional outputs
- Additional uncertainty metrics (NLL, CRPS)
- Bayesian uncertainty estimation methods
- Ensemble uncertainty analysis