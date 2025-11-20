#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from alp.data.data_reading import force_grid_endpoints
from alp.networks.mlp import MLP
from alp.networks.losses import heteroscedastic_gaussian_nll
from alp.utils.logger_config import logger


def main(heteroscedastic=False):

    here = os.path.dirname(__file__)
    data_file = os.path.join(here, '..', '..', 'data', 'Hz31.txt')

    # -------- 1) Load data --------
    data = pd.read_csv(
        data_file, names=['z', 'hz', 'err'], sep=r"\s+", engine='python'
    )
    z_raw  = data['z'].values.astype(float)
    hz_raw = data['hz'].values.astype(float)
    err_raw = data['err'].values.astype(float)

    # -------- 2) Build training grid & force endpoints --------
    zmin, zmax = float(np.min(z_raw)), float(np.max(z_raw))
    train_grid = np.linspace(zmin, zmax, 100)

    z_adj, hz_adj   = force_grid_endpoints(z_raw,  hz_raw,  train_grid)
    _,    err_adj  = force_grid_endpoints(z_raw,  err_raw, train_grid)

    order = np.argsort(z_adj)
    z_adj, hz_adj, err_adj = z_adj[order], hz_adj[order], err_adj[order]

    # -------- 3) Split data (keep endpoints in train) --------
    z_min, z_max = train_grid[0], train_grid[-1]
    i_min = np.where(z_adj == z_min)[0][0]
    i_max = np.where(z_adj == z_max)[0][0]
    keep_idx = {i_min, i_max}

    all_idx = np.arange(len(z_adj))
    rest_idx = np.array([i for i in all_idx if i not in keep_idx])

    z_rest, hz_rest, err_rest = z_adj[rest_idx], hz_adj[rest_idx], err_adj[rest_idx]

    z_tr, z_tmp, y_tr, y_tmp, e_tr, e_tmp = train_test_split(
        z_rest, hz_rest, err_rest, test_size=0.30, random_state=42
    )
    z_va, z_te, y_va, y_te, e_va, e_te = train_test_split(
        z_tmp, y_tmp, e_tmp, test_size=0.50, random_state=42
    )

    # add endpoints to training set
    z_tr = np.concatenate([z_tr, [z_min, z_max]])
    y_tr = np.concatenate([y_tr, [hz_adj[i_min], hz_adj[i_max]]])
    e_tr = np.concatenate([e_tr, [err_adj[i_min], err_adj[i_max]]])

    # -------- 4) Scale z with training set only --------
    scalerz = StandardScaler()
    X_tr = scalerz.fit_transform(z_tr.reshape(-1, 1))
    X_va = scalerz.transform(z_va.reshape(-1, 1))
    X_te = scalerz.transform(z_te.reshape(-1, 1))

    Y_tr = y_tr.reshape(-1, 1)
    Y_va = y_va.reshape(-1, 1)
    Y_te = y_te.reshape(-1, 1)

    # sample weights (only used in non-heteroscedastic mode)
    w_tr = 1.0 / np.clip(e_tr, 1e-8, np.inf)**2
    w_va = 1.0 / np.clip(e_va, 1e-8, np.inf)**2

    # -------- 5) Build ALP MLP --------
    tf.keras.utils.set_random_seed(123)

    net = MLP(
        n_inputs=1,
        deep=[64, 64, 64],
        actfn='relu',
        dropout=0.1,
        mcdropout=True, 
        n_outputs=2
    )
    model = net.model_tf()

    if heteroscedastic:
        loss_fn = heteroscedastic_gaussian_nll
        sample_weight_tr = None
        val_data = (X_va, Y_va)
    else:
        # classic MSE with 1/err^2 weights -> closer to previous papers
        def mse_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
        loss_fn = mse_loss
        sample_weight_tr = w_tr
        val_data = (X_va, Y_va, w_va)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn
    )

    logger.info("Starting training (heteroscedastic=%s)...", heteroscedastic)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=25, factor=0.5,
            min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50,
            restore_best_weights=True, verbose=1
        )
    ]

    history = model.fit(
        X_tr, Y_tr,
        sample_weight=sample_weight_tr,
        validation_data=val_data,
        epochs=500,
        batch_size=32,
        verbose=2,
        callbacks=callbacks
    )

    # -------- 6) MC-Dropout predictions --------
    def mcdo_predict(m, X, n_samples=200):
        preds = []
        for _ in range(n_samples):
            preds.append(m(X, training=True).numpy())
        P = np.stack(preds, axis=0)  # [S, N, out_dim]
        mean = P.mean(axis=0)
        std  = P.std(axis=0)
        return mean, std

    y_mu_te, y_std_te = mcdo_predict(model, X_te)

    if heteroscedastic:
        # y_mu_te[:, 0] -> mean y
        # y_mu_te[:, 1] -> mean log_sigma2
        y_pred = y_mu_te[:, 0]
        log_sigma2_pred = y_mu_te[:, 1]
        sigma_aleatoric = np.sqrt(np.exp(log_sigma2_pred))
        sigma_epistemic = y_std_te[:, 0]
        sigma_total = np.sqrt(sigma_aleatoric**2 + sigma_epistemic**2)
    else:
        y_pred = y_mu_te[:, 0]
        sigma_total = y_std_te[:, 0]  # epistemic only

    rmse_te = float(np.sqrt(np.mean((y_pred - Y_te[:, 0])**2)))
    logger.info(f"Test RMSE: {rmse_te:.3f}")

    outdir = os.path.join(here, 'outputs')
    os.makedirs(outdir, exist_ok=True)

    np.savetxt(
        os.path.join(outdir, f'test_predictions_hetero_{heteroscedastic}.txt'),
        np.c_[z_te, Y_te[:, 0], y_pred, sigma_total],
        header="z  H(z)_true  H(z)_pred  sigma_total"
    )
    model.save(os.path.join(outdir, f'alp_mlp_hz_hetero_{heteroscedastic}'))
    logger.info("Done.")


if __name__ == "__main__":
    # 1) reproduce previous results:
    main()
    # 2) then run heteroscedastic version:
    # main(heteroscedastic=True)
