#!/usr/bin/env python
"""Cosmological Functions for ALP Framework

This module provides standard cosmological calculations including
distance modulus, comoving distance, and Hubble parameter
calculations for various dark energy models.
"""

import numpy as np
from scipy import integrate
from typing import Union, Tuple


class Cosmology:
    """Standard cosmological calculations for various dark energy models."""

    def __init__(self, H0: float = 70.0, Om: float = 0.27, w0: float = -1.0, wa: float = 0.0):
        """Initialize cosmological parameters.

        Args:
            H0 (float): Hubble constant in km/s/Mpc (default: 70.0)
            Om (float): Matter density parameter Ω_m (default: 0.27)
            w0 (float): Dark energy equation of state parameter w_0 (default: -1.0)
            wa (float): Dark energy evolution parameter w_a (default: 0.0)
        """
        self.H0 = H0
        self.Om = Om
        self.w0 = w0
        self.wa = wa
        self.c = 299792.458  # Speed of light in km/s

    def rhsquared_a_owacdm(self, a: float) -> float:
        """Calculate H²(a)/H₀² for w₀wₐCDM model.

        Args:
            a (float): Scale factor

        Returns:
            float: H²(a)/H₀²
        """
        rhow = a ** (-3 * (1.0 + self.w0 + self.wa)) * np.exp(-3 * self.wa * (1 - a))
        return self.Om / a**3 + (1.0 - self.Om) * rhow

    def dist_integrand_a(self, a: float) -> float:
        """Integrand for comoving distance calculation.

        Args:
            a (float): Scale factor

        Returns:
            float: Integrand value
        """
        return 1.0 / np.sqrt(self.rhsquared_a_owacdm(a)) / a**2

    def da_z(self, z: float) -> float:
        """Calculate comoving distance D_a(z) in Mpc/h.

        Args:
            z (float): Redshift

        Returns:
            float: Comoving distance in Mpc/h
        """
        result = integrate.quad(self.dist_integrand_a, 1.0 / (1 + z), 1, epsabs=1e-8, epsrel=1e-8)
        return result[0]

    def distance_modulus(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate distance modulus μ(z) for ΛCDM/w₀wₐCDM model.

        Uses proper cosmological constants and unit conversions to ensure
        consistency with observational data (μ ≈ 34-45 for LSST).

        Args:
            z (float or np.ndarray): Redshift

        Returns:
            float or np.ndarray: Distance modulus μ(z)
        """
        # Handle array input
        z_array = np.array(z) if isinstance(z, (list, np.ndarray)) else z

        if isinstance(z_array, np.ndarray):
            return np.array([self.distance_modulus(z_val) for z_val in z_array])

        # Calculate comoving distance
        da = self.da_z(z_array)

        # Distance modulus: μ = 5*log10(d_L/10pc)
        # where d_L = (c/H₀) * D_a * (1+z) is the luminosity distance
        # Using proper cosmological constants:
        # c = 299792.458 km/s (speed of light)
        # H₀ = 70.0 km/s/Mpc (Hubble constant)
        lum_distance = (self.c / self.H0) * da * (1 + z_array)

        # Convert to distance modulus and add constant for proper scale
        mu = 5 * np.log10(lum_distance) + 25

        return mu


# Convenience functions for backward compatibility
def create_lcdm_model(H0: float = 70.0, Om: float = 0.27) -> Cosmology:
    """Create standard ΛCDM cosmology model.

    Args:
        H0 (float): Hubble constant in km/s/Mpc
        Om (float): Matter density parameter

    Returns:
        Cosmology: ΛCDM cosmology object
    """
    return Cosmology(H0=H0, Om=Om, w0=-1.0, wa=0.0)


def create_wacdm_model(
    H0: float = 70.0, Om: float = 0.27, w0: float = -1.0, wa: float = 0.0
) -> Cosmology:
    """Create w₀wₐCDM cosmology model.

    Args:
        H0 (float): Hubble constant in km/s/Mpc
        Om (float): Matter density parameter
        w0 (float): Dark energy equation of state parameter w₀
        wa (float): Dark energy evolution parameter wₐ

    Returns:
        Cosmology: w₀wₐCDM cosmology object
    """
    return Cosmology(H0=H0, Om=Om, w0=w0, wa=wa)


def calculate_distance_modulus_range(
    z_min: float = 0.01, z_max: float = 2.4, n_points: int = 100, cosmology: Cosmology = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate distance modulus over a redshift range.

    Args:
        z_min (float): Minimum redshift
        z_max (float): Maximum redshift
        n_points (int): Number of points to calculate
        cosmology (Cosmology): Cosmology model (uses ΛCDM if None)

    Returns:
        tuple: (z_array, mu_array)
    """
    if cosmology is None:
        cosmology = create_lcdm_model()

    z_array = np.linspace(z_min, z_max, n_points)
    mu_array = cosmology.distance_modulus(z_array)

    return z_array, mu_array


def get_lcdm_hubble(z_range: np.ndarray, cosmology: Cosmology = None) -> np.ndarray:
    """Get ΛCDM Hubble parameter predictions.

    Parameters
    ----------
    z_range : np.ndarray
        Redshift values
    cosmology : Cosmology, optional
        Cosmology model (uses ΛCDM if None)

    Returns
    -------
    np.ndarray
        H(z) values in km/s/Mpc for the given redshifts
    """
    if cosmology is None:
        cosmology = create_lcdm_model()

    hz = []
    for z in z_range:
        a = 1.0 / (1.0 + z)
        hz.append(cosmology.H0 * np.sqrt(cosmology.rhsquared_a_owacdm(a)))
    return np.array(hz)
