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


def force_x_range_in_training(X_train, y_train, X_all, y_all):
    """
    Ensure that the samples containing the minimum and maximum values
    of each input feature in the full dataset are included in the
    training set.

    This is useful when splitting data into train/test sets to ensure
    the model sees the full range of inputs during training.

    Parameters
    ----------
    X_train : array-like
        Training input features (2D array with samples as rows).
    y_train : array-like
        Training target values corresponding to X_train.
    X_all : array-like
        Full input features (e.g., before split) to determine min/max range.
    y_all : array-like
        Full target values corresponding to X_all.

    Returns
    -------
    X_train_new, y_train_new : np.ndarray
        Updated training arrays including the min/max samples if they were missing.
    added_indices : list
        Indices of samples added to training set (for handling auxiliary arrays like errors).
    """
    X_train = (
        np.array(X_train, dtype=float).reshape(-1, 1)
        if np.array(X_train).ndim == 1
        else np.array(X_train, dtype=float)
    )
    y_train = (
        np.array(y_train, dtype=float).reshape(-1, 1)
        if np.array(y_train).ndim == 1
        else np.array(y_train, dtype=float)
    )
    X_all = (
        np.array(X_all, dtype=float).reshape(-1, 1)
        if np.array(X_all).ndim == 1
        else np.array(X_all, dtype=float)
    )
    y_all = (
        np.array(y_all, dtype=float).reshape(-1, 1)
        if np.array(y_all).ndim == 1
        else np.array(y_all, dtype=float)
    )

    X_new = X_train.copy()
    y_new = y_train.copy()
    added_indices = []

    n_features = X_all.shape[1]

    for i in range(n_features):
        min_idx = np.argmin(X_all[:, i])
        max_idx = np.argmax(X_all[:, i])

        min_in_train = np.any(np.all(np.isclose(X_train, X_all[min_idx]), axis=1))
        max_in_train = np.any(np.all(np.isclose(X_train, X_all[max_idx]), axis=1))

        if not min_in_train:
            X_new = np.vstack([X_new, X_all[min_idx]])
            y_new = np.append(y_new, y_all[min_idx])
            added_indices.append(min_idx)

        if not max_in_train and max_idx != min_idx:
            X_new = np.vstack([X_new, X_all[max_idx]])
            y_new = np.append(y_new, y_all[max_idx])
            added_indices.append(max_idx)

    return X_new, y_new, added_indices
