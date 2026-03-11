#!/usr/bin/env python
"""ALP Dataset Loading Utilities"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_lsst_data(data_path="data/lsst_hubble_diagram.txt"):
    """Load LSST supernova data for dual-output regression.

    Args:
        data_path (str): Path to LSST hubble_diagram.txt file

    Returns:
        tuple: (z_data, mu_data, error_data)
            - z_data: Redshift values (N, 1)
            - mu_data: Distance modulus values (N, 1)
            - error_data: Combined errors (N, 1)
    """
    df_data = pd.read_csv(data_path, skiprows=5, sep=" ")
    df_data["errors"] = df_data["MUERR"].values + df_data["MUERR_SYS"].values
    df_data = df_data[["zCMB", "MU", "errors"]]

    z_data = df_data[["zCMB"]].values
    mu_data = df_data[["MU"]].values
    error_data = df_data[["errors"]].values

    return z_data, mu_data, error_data


def preprocess_lsst_data(z_data, mu_data, error_data, train_split=0.8, random_state=42):
    """Preprocess LSST data with scaling and train/test split.

    Args:
        z_data (np.array): Redshift values
        mu_data (np.array): Distance modulus values
        error_data (np.array): Error values
        train_split (float): Training split ratio
        random_state (int): Random seed

    Returns:
        tuple: (z_train, z_test, y_train, y_test, scaler)
    """
    # Ensure we're working with numpy arrays
    z_data = np.array(z_data).flatten()
    mu_data = np.array(mu_data).flatten()
    error_data = np.array(error_data).flatten()

    # Combine mu and error for dual output
    y_data = np.column_stack([mu_data, error_data])

    # Randomize
    randomize = np.random.RandomState(random_state).permutation(len(z_data))
    z_data = z_data[randomize]
    y_data = y_data[randomize]

    # Scale input (only z, not mu - we want network to learn actual mu scale)
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_data.reshape(-1, 1))

    # Train/test split
    n_train = int(train_split * len(z_scaled))
    z_train, z_test = np.split(z_scaled, [n_train])
    y_train, y_test = np.split(y_data, [n_train])

    return z_train, z_test, y_train, y_test, scaler


def load_hz31_data(data_path="data/Hz31.txt"):
    """Load H(z) data for dual-output regression.

    Args:
        data_path (str): Path to Hz31.txt file

    Returns:
        tuple: (z_data, hz_data, error_data)
    """
    data = pd.read_csv(data_path, names=["z", "hz", "err"], sep=r"\s+", engine="python")
    z_data = data[["z"]].values
    hz_data = data[["hz"]].values
    error_data = data[["err"]].values

    return z_data, hz_data, error_data


def preprocess_cc_data(z_data, hz_data, error_data, random_state=42):
    """Preprocess CC (Cosmic Chronometers) data with proper train/test split.

    Uses StandardScaler for input scaling and maintains grid endpoint integrity.
    Prepares data in dual-output format [H(z), error].

    Args:
        z_data (np.array): Redshift values
        hz_data (np.array): H(z) values
        error_data (np.array): Error values
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (z_train, z_test, y_train, y_test, scaler)
            - z_train: Scaled training redshifts (N_train, 1)
            - z_test: Scaled test redshifts (N_test, 1)
            - y_train: Dual-output training targets (N_train, 2) with [H(z), error]
            - y_test: Dual-output test targets (N_test, 2) with [H(z), error]
            - scaler: StandardScaler fitted on training data for later predictions
    """
    from alp.data.data_reading import force_grid_endpoints, force_x_range_in_training

    # Flatten input data
    z_raw = np.array(z_data).flatten()
    hz_raw = np.array(hz_data).flatten()
    err_raw = np.array(error_data).flatten()

    # Force grid endpoints to ensure boundary coverage
    zmin, zmax = float(np.min(z_raw)), float(np.max(z_raw))
    train_grid = np.linspace(zmin, zmax, 100)

    z_adj, hz_adj = force_grid_endpoints(z_raw, hz_raw, train_grid)
    _, err_adj = force_grid_endpoints(z_raw, err_raw, train_grid)

    # Sort by redshift
    order = np.argsort(z_adj)
    z_adj, hz_adj, err_adj = z_adj[order], hz_adj[order], err_adj[order]

    # Identify and preserve grid endpoints
    z_min, z_max = train_grid[0], train_grid[-1]
    i_min = np.where(z_adj == z_min)[0][0]
    i_max = np.where(z_adj == z_max)[0][0]
    keep_idx = {i_min, i_max}

    all_idx = np.arange(len(z_adj))
    rest_idx = np.array([i for i in all_idx if i not in keep_idx])

    z_rest, hz_rest, err_rest = z_adj[rest_idx], hz_adj[rest_idx], err_adj[rest_idx]

    # Split data (70% train, 30% temp -> 50/50 split of temp into val/test)
    z_tr, z_tmp, y_tr, y_tmp, e_tr, e_tmp = train_test_split(
        z_rest, hz_rest, err_rest, test_size=0.30, random_state=random_state
    )
    z_va, z_te, y_va, y_te, e_va, e_te = train_test_split(
        z_tmp, y_tmp, e_tmp, test_size=0.50, random_state=random_state
    )

    # Add grid endpoints to training set
    z_tr = np.concatenate([z_tr, [z_min, z_max]])
    y_tr = np.concatenate([y_tr, [hz_adj[i_min], hz_adj[i_max]]])
    e_tr = np.concatenate([e_tr, [err_adj[i_min], err_adj[i_max]]])

    # Force x-range in training (ensures training bounds are respected)
    n_train_before = len(z_tr)
    X_all = np.concatenate([z_tr, z_va, z_te])
    y_all = np.concatenate([y_tr, y_va, y_te])
    z_tr, y_tr, added_indices = force_x_range_in_training(z_tr, y_tr, X_all, y_all)

    if added_indices:
        e_all = np.concatenate([e_tr[:n_train_before], e_va, e_te])
        new_errors = np.array([e_all[i] for i in added_indices])
        e_tr = np.concatenate([e_tr, new_errors])

    # Scale inputs only (z values)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_tr.reshape(-1, 1))
    z_test = scaler.transform(z_te.reshape(-1, 1))

    # Combine H(z) and error into dual-output target
    y_train = np.column_stack([y_tr, e_tr])
    y_test = np.column_stack([y_te, e_te])

    return z_train, z_test, y_train, y_test, scaler


def load_pantheon_data(data_path="data/pantheon_lcparam_full_long_zhel.txt"):
    """Load Pantheon supernova data for dual-output regression.

    Args:
        data_path (str): Path to pantheon_lcparam_full_long_zhel.txt file

    Returns:
        tuple: (z_data, mb_data, dmb_data)
            - z_data: Heliocentric redshift values (N, 1)
            - mb_data: Apparent magnitude values (N, 1)
            - dmb_data: Magnitude errors (N, 1)
    """
    # Read the header line and extract column names
    with open(data_path, 'r') as f:
        header_line = f.readline().strip()
    # Remove the '#' character if present
    if header_line.startswith('#'):
        header_line = header_line[1:]
    col_names = header_line.split()
    
    # Read data with explicit column names
    df_data = pd.read_csv(data_path, skiprows=1, sep=r"\s+", engine="python", names=col_names)
    z_data = df_data[["zhel"]].values
    mb_data = df_data[["mb"]].values
    dmb_data = df_data[["dmb"]].values

    return z_data, mb_data, dmb_data


def preprocess_pantheon_data(z_data, mb_data, dmb_data, train_split=0.8, random_state=42):
    """Preprocess Pantheon data with scaling and train/test split.

    Args:
        z_data (np.array): Redshift values
        mb_data (np.array): Apparent magnitude values
        dmb_data (np.array): Magnitude error values
        train_split (float): Training split ratio
        random_state (int): Random seed

    Returns:
        tuple: (z_train, z_test, y_train, y_test, scaler)
    """
    z_data = np.array(z_data).flatten()
    mb_data = np.array(mb_data).flatten()
    dmb_data = np.array(dmb_data).flatten()

    # Combine mb and error for dual output
    y_data = np.column_stack([mb_data, dmb_data])

    # Randomize
    randomize = np.random.RandomState(random_state).permutation(len(z_data))
    z_data = z_data[randomize]
    y_data = y_data[randomize]

    # Scale input
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_data.reshape(-1, 1))

    # Train/test split
    n_train = int(train_split * len(z_scaled))
    z_train, z_test = np.split(z_scaled, [n_train])
    y_train, y_test = np.split(y_data, [n_train])

    return z_train, z_test, y_train, y_test, scaler


def load_fs8_data(data_path="data/fs8Diagram.txt"):
    """Load fs8 growth rate data for dual-output regression.

    Args:
        data_path (str): Path to fs8Diagram.txt file

    Returns:
        tuple: (z_data, fs8_data, error_data)
            - z_data: Redshift values (N, 1)
            - fs8_data: fs8 values (N, 1)
            - error_data: Error values (N, 1)
    """
    data = pd.read_csv(data_path, skiprows=3, names=["z", "fs8", "error", "om_ref"],
                       sep=r"\t", engine="python")
    z_data = data[["z"]].values
    fs8_data = data[["fs8"]].values
    error_data = data[["error"]].values

    return z_data, fs8_data, error_data


def preprocess_fs8_data(z_data, fs8_data, error_data, train_split=0.8, random_state=42):
    """Preprocess fs8 data with scaling and train/test split.

    Args:
        z_data (np.array): Redshift values
        fs8_data (np.array): fs8 values
        error_data (np.array): Error values
        train_split (float): Training split ratio
        random_state (int): Random seed

    Returns:
        tuple: (z_train, z_test, y_train, y_test, scaler)
    """
    z_data = np.array(z_data).flatten()
    fs8_data = np.array(fs8_data).flatten()
    error_data = np.array(error_data).flatten()

    # Combine fs8 and error for dual output
    y_data = np.column_stack([fs8_data, error_data])

    # Randomize
    randomize = np.random.RandomState(random_state).permutation(len(z_data))
    z_data = z_data[randomize]
    y_data = y_data[randomize]

    # Scale input
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_data.reshape(-1, 1))

    # Train/test split
    n_train = int(train_split * len(z_scaled))
    z_train, z_test = np.split(z_scaled, [n_train])
    y_train, y_test = np.split(y_data, [n_train])

    return z_train, z_test, y_train, y_test, scaler


def load_des_data(data_path="data/DES-SN5YR_HD.csv"):
    """Load DES supernova data for dual-output regression.

    Args:
        data_path (str): Path to DES-SN5YR_HD.csv file

    Returns:
        tuple: (z_data, mu_data, muerr_data)
            - z_data: Heliocentric redshift values (N, 1)
            - mu_data: Distance modulus values (N, 1)
            - muerr_data: Distance modulus errors (N, 1)
    """
    df_data = pd.read_csv(data_path)
    z_data = df_data[["zHEL"]].values
    mu_data = df_data[["MU"]].values
    muerr_data = df_data[["MUERR_FINAL"]].values

    return z_data, mu_data, muerr_data


def preprocess_des_data(z_data, mu_data, muerr_data, train_split=0.8, random_state=42):
    """Preprocess DES data with scaling and train/test split.

    Args:
        z_data (np.array): Redshift values
        mu_data (np.array): Distance modulus values
        muerr_data (np.array): Distance modulus error values
        train_split (float): Training split ratio
        random_state (int): Random seed

    Returns:
        tuple: (z_train, z_test, y_train, y_test, scaler)
    """
    z_data = np.array(z_data).flatten()
    mu_data = np.array(mu_data).flatten()
    muerr_data = np.array(muerr_data).flatten()

    # Combine mu and error for dual output
    y_data = np.column_stack([mu_data, muerr_data])

    # Randomize
    randomize = np.random.RandomState(random_state).permutation(len(z_data))
    z_data = z_data[randomize]
    y_data = y_data[randomize]

    # Scale input
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_data.reshape(-1, 1))

    # Train/test split
    n_train = int(train_split * len(z_scaled))
    z_train, z_test = np.split(z_scaled, [n_train])
    y_train, y_test = np.split(y_data, [n_train])

    return z_train, z_test, y_train, y_test, scaler
