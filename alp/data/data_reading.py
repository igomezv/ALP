#!/usr/bin/env python

"""
ALP Data Reading Utilities
--------------------------
Methods for loading observational datasets (e.g., H(z)) and forcing
consistency with the training redshift grid: specifically ensuring that the
first and last redshifts from the training set are present in each dataset.
"""

import numpy as np


def force_grid_endpoints(dataset_z, dataset_y, train_z):
    """
    Ensure that the dataset contains the first and last z-points
    from the training grid. If missing, they are inserted via
    linear interpolation/extrapolation.

    Parameters
    ----------
    dataset_z : array-like
        Observational redshift values (sorted or unsorted).
    dataset_y : array-like
        Observational measurements corresponding to dataset_z.
    train_z : array-like
        Training redshift grid for the ML model.

    Returns
    -------
    z_new, y_new : np.ndarray
        Updated arrays including the endpoints (if they were missing).
    """

    dataset_z = np.array(dataset_z, dtype=float)
    dataset_y = np.array(dataset_y, dtype=float)
    train_z = np.array(train_z, dtype=float)

    # Sort dataset if needed
    order = np.argsort(dataset_z)
    dataset_z = dataset_z[order]
    dataset_y = dataset_y[order]

    z_min, z_max = train_z[0], train_z[-1]

    new_z = dataset_z.copy()
    new_y = dataset_y.copy()

    # ---- FORCE FIRST POINT ----
    if z_min not in new_z:
        # Extrapolate/interpolate using numpy interp
        y_first = np.interp(z_min, new_z, new_y)
        new_z = np.insert(new_z, 0, z_min)
        new_y = np.insert(new_y, 0, y_first)

    # ---- FORCE LAST POINT ----
    if z_max not in new_z:
        y_last = np.interp(z_max, new_z, new_y)
        new_z = np.append(new_z, z_max)
        new_y = np.append(new_y, y_last)

    # Re-sort just in case
    order = np.argsort(new_z)
    return new_z[order], new_y[order]

