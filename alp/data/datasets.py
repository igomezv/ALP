#!/usr/bin/env python
"""ALP Dataset Loading Utilities"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_lsst_data(data_path='Data_SNIa_LSST/larger/hubble_diagram.txt'):
    """Load LSST supernova data for dual-output regression.
    
    Args:
        data_path (str): Path to LSST hubble_diagram.txt file
        
    Returns:
        tuple: (z_data, mu_data, error_data)
            - z_data: Redshift values (N, 1)
            - mu_data: Distance modulus values (N, 1) 
            - error_data: Combined errors (N, 1)
    """
    df_data = pd.read_csv(data_path, skiprows=5, sep=' ')
    df_data['errors'] = df_data['MUERR'].values + df_data['MUERR_SYS'].values
    df_data = df_data[['zCMB', 'MU', 'errors']]
    
    z_data = df_data[['zCMB']].values
    mu_data = df_data[['MU']].values
    error_data = df_data[['errors']].values
    
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

def load_hz31_data(data_path='data/Hz31.txt'):
    """Load H(z) data for dual-output regression.
    
    Args:
        data_path (str): Path to Hz31.txt file
        
    Returns:
        tuple: (z_data, hz_data, error_data)
    """
    data = pd.read_csv(
        data_path, names=['z', 'hz', 'err'], sep=r"\s+", engine='python'
    )
    z_data = data[['z']].values
    hz_data = data[['hz']].values
    error_data = data[['err']].values
    
    return z_data, hz_data, error_data