#!/usr/bin/env python
"""Visualization Utilities for ALP Results"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..utils.logger_config import logger

def plot_lsst_results(
    z_test_range: np.ndarray,
    results: dict,
    z_data: np.ndarray,
    mu_data: np.ndarray,
    error_data: np.ndarray,
    history: Optional[dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 6),
    dpi: int = 100
) -> None:
    """Plot LSST dual-output regression results.
    
    Args:
        z_test_range: Test redshift range for predictions
        results: Dictionary with 'mean', 'std', 'combined_uncertainty'
        z_data: Training redshift data
        mu_data: Training distance modulus data
        error_data: Training error data
        history: Training history dictionary (optional)
        save_path: Path to save plot (optional)
        figsize: Figure size tuple
        dpi: Figure DPI
    """
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        if history is not None:
            # Training history subplot
            ax1 = plt.subplot(1, 3, 1)
            ax1.plot(history['loss'], 'r', label='Training', linewidth=2)
            ax1.plot(history['val_loss'], 'g', label='Validation', linewidth=2)
            ax1.set_ylabel('MSE', fontsize=12)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.set_title('Training History', fontsize=12)
            ax1.grid(True, alpha=0.3)
            if 'loss' in history:
                max_loss = max(max(history['loss']), max(history['val_loss']))
                ax1.set_ylim(0, max_loss * 1.1)
        
        # Main results subplot
        ax2 = plt.subplot(1, 2, (2 if history is None else 2))
        
        # Plot observations
        ax2.errorbar(
            z_data.flatten(), mu_data.flatten(), error_data.flatten(), 
            fmt='g.', markersize=2, alpha=0.4, capsize=2, 
            label='LSST Observations', ecolor='green', elinewidth=0.5
        )
        
        # Plot ALP predictions with uncertainty
        ax2.errorbar(
            z_test_range, results['mean'][:, 0]-19, 
            results['combined_uncertainty'],
            markersize=3, fmt='o', ecolor='red', capthick=2, 
            elinewidth=1, alpha=0.7, c='magenta',
            label='ALP Predictions ± σ'
        )
        
        ax2.set_xlabel('Redshift z', fontsize=14)
        ax2.set_ylabel('Distance Modulus μ(z)', fontsize=14)
        ax2.set_title('LSST Distance Modulus Prediction', fontsize=14)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting LSST results: {e}")

def plot_training_history(history: dict, save_path: Optional[str] = None) -> None:
    """Plot training history with loss curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    try:
        plt.figure(figsize=(8, 6))
        
        epochs = range(1, len(history['loss']) + 1)
        plt.plot(epochs, history['loss'], 'r', label='Training Loss', linewidth=2)
        plt.plot(epochs, history['val_loss'], 'g', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training History', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limit for better visualization
        max_loss = max(max(history['loss']), max(history['val_loss']))
        plt.ylim(0, max_loss * 1.1)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")

def plot_uncertainty_analysis(
    z_test: np.ndarray,
    results: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100
) -> None:
    """Plot uncertainty analysis showing epistemic vs aleatoric components.
    
    Args:
        z_test: Test redshift values
        results: Dictionary with uncertainty components
        save_path: Path to save plot (optional)
        figsize: Figure size tuple
        dpi: Figure DPI
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Mean prediction
        axes[0, 0].plot(z_test, results['mean'][:, 0], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Redshift z')
        axes[0, 0].set_ylabel('Distance Modulus μ(z)')
        axes[0, 0].set_title('Mean Prediction')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Epistemic uncertainty (MC dropout std)
        axes[0, 1].plot(z_test, results['std'][:, 0], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Redshift z')
        axes[0, 1].set_ylabel('Epistemic Uncertainty σ')
        axes[0, 1].set_title('Epistemic Uncertainty (MC Dropout)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Aleatoric uncertainty (predicted error)
        axes[1, 0].plot(z_test, results['mean'][:, 1], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Redshift z')
        axes[1, 0].set_ylabel('Aleatoric Uncertainty σ')
        axes[1, 0].set_title('Aleatoric Uncertainty (Predicted Error)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined uncertainty
        axes[1, 1].plot(z_test, results['combined_uncertainty'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Redshift z')
        axes[1, 1].set_ylabel('Combined Uncertainty σ')
        axes[1, 1].set_title('Combined Uncertainty (Total)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Uncertainty analysis plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting uncertainty analysis: {e}")

def create_lcdm_model():
    """Create ΛCDM model for comparison.
    
    Returns:
        tuple: (z_model, flcdm) - redshift range and distance modulus
    """
    try:
        from scipy import integrate
        
        def RHSquared_a_owacdm(a, w0, wa, Om):
            rhow = a**(-3*(1.0+w0+wa))*np.exp(-3*wa*(1-a))
            return (Om/a**3+(1.0-Om)*rhow)
        
        def DistIntegrand_a(a, w0, wa, Om):
            return 1./np.sqrt(RHSquared_a_owacdm(a, w0, wa, Om))/a**2
        
        def Da_z(z, w0, wa, Om):
            r = integrate.quad(DistIntegrand_a, 1./(1+z), 1, args=(w0, wa, Om))
            return r[0]
        
        def distance_modulus(z, w0=-1, wa=0.0, Om=0.27):
            return 5*np.log10(Da_z(z, w0, wa, Om)*(1+z))+24
        
        z_model = np.linspace(0.01, 2.4, 100)
        flcdm = np.array([distance_modulus(zzz, w0=-1, wa=0, Om=0.27) for zzz in z_model])
        
        return z_model, flcdm
        
    except ImportError:
        logger.warning("Scipy not available for ΛCDM model generation")
        return None, None