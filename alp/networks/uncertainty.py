#!/usr/bin/env python
"""Uncertainty Quantification for ALP Networks"""

import numpy as np
from typing import Dict, Any, Optional
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from ..utils.logger_config import logger

class UncertaintyQuantifier:
    """Handles MC Dropout uncertainty quantification for dual-output models.
    
    Performs Monte Carlo Dropout inference to estimate epistemic uncertainty
    and combines it with aleatoric uncertainty for comprehensive uncertainty quantification.
    """
    
    def __init__(self, n_samples: int = 100):
        """Initialize uncertainty quantifier.
        
        Args:
            n_samples (int): Number of MC dropout forward passes
        """
        self.n_samples = n_samples
        logger.info(f"UncertaintyQuantifier initialized with {n_samples} MC samples")
    
    def mc_dropout_prediction(
        self, 
        model, 
        X, 
        scaler: Optional[StandardScaler] = None
    ) -> Dict[str, Any]:
        """Perform MC Dropout prediction with uncertainty quantification.
        
        Args:
            model: Trained ALP model with MC Dropout
            X: Input data (N, 1) or (N,)
            scaler: Input scaler (if needed)
            
        Returns:
            dict: {
                'mean': Mean predictions (N, 2),
                'std': Standard deviations (N, 2),
                'combined_uncertainty': Combined uncertainty for Y1 (N,)
            }
        """
        if scaler is not None:
            X = scaler.transform(X.reshape(-1, 1))
        elif len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        logger.info(f"Performing {self.n_samples} MC dropout predictions...")
        
        predictions = []
        for i in range(self.n_samples):
            if (i + 1) % 20 == 0:
                logger.info(f"  MC prediction {i + 1}/{self.n_samples}")
            # Use the model directly with training=True for MC Dropout
            pred = model(X, training=True).numpy()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0, ddof=1)
        
        # Combined uncertainty for Y1 (distance modulus)
        # σ_total = sqrt(σ_epistemic^2 + σ_aleatoric^2)
        # where σ_epistemic = std_pred[:, 0] (MC dropout std for Y1)
        # and σ_aleatoric = std_pred[:, 1] (predicted error)
        combined_uncertainty = np.sqrt(
            std_pred[:, 0]**2 + std_pred[:, 1]**2 + mean_pred[:, 1]**2
        )
        
        logger.info("MC dropout predictions completed successfully")
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'combined_uncertainty': combined_uncertainty
        }
    
    def calculate_confidence_intervals(
        self, 
        mean: np.ndarray, 
        uncertainty: np.ndarray, 
        confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions.
        
        Args:
            mean: Mean predictions
            uncertainty: Standard deviations
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            dict: {
                'lower': Lower bound of confidence interval,
                'upper': Upper bound of confidence interval,
                'z_score': Z-score for confidence level
            }
        """
        from scipy import stats
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate confidence intervals
        margin = z_score * uncertainty
        lower_bound = mean - margin
        upper_bound = mean + margin
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'z_score': z_score
        }
    
    def decompose_uncertainty(
        self, 
        mc_dropout_std: np.ndarray, 
        predicted_error: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Decompose total uncertainty into epistemic and aleatoric components.
        
        Args:
            mc_dropout_std: Standard deviation from MC dropout (epistemic)
            predicted_error: Predicted error (aleatoric)
            
        Returns:
            dict: {
                'epistemic': Epistemic uncertainty,
                'aleatoric': Aleatoric uncertainty,
                'total': Total combined uncertainty
            }
        """
        epistemic = mc_dropout_std
        aleatoric = predicted_error
        total = np.sqrt(epistemic**2 + aleatoric**2)
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }