# Pytorch/trainers/pca_trainer.py
import os
import numpy as np
import matplotlib.pyplot as plt
from Backend.app import db
from Backend.app.models.latent_models import LatentPoint, PCAProjectionConfig, PCAProjection
from Pytorch.projections.pca import fit_pca, plot_explained_variance, plot_principal_components


def run_pca_for_run(session, run_id, hyperparams, plot_dir=None):
    """
    Train PCA on latent vectors for a given model run,
    persist the results in the database, and optionally save diagnostic plots.

    :param session: SQLAlchemy session
    :param run_id: ID of the ModelRun to process
    :param hyperparams: dict containing PCA hyperparameters:
        - n_components (int)
        - additional_params (dict)
    :param plot_dir: optional path to save plots; if None, no files are written
    :returns: id of created PCAProjectionConfig
    """
    # 1) Fetch latent points
    points = (
        session.query(LatentPoint)
        .filter_by(model_run_id=run_id)
        .order_by(LatentPoint.id)
        .all()
    )
    if not points:
        raise ValueError(f"No LatentPoint entries found for run_id={run_id}")

    Z = np.vstack([p.latent_vector for p in points])

    # 2) Read hyperparameters
    n_components = hyperparams.get('n_components', min(Z.shape))
    additional = hyperparams.get('additional_params', {})

    # 3) Fit PCA
    pca, X_reduced = fit_pca(Z, n_components, additional)

    # 4) Persist PCAProjectionConfig
    pca_cfg = PCAProjectionConfig(
        model_run_id=run_id,
        n_components=n_components,
        additional_params=additional
    )
    session.add(pca_cfg)
    session.flush()

    # 5) Persist PCAProjection rows
    for idx, vec in enumerate(X_reduced):
        for dim, val in enumerate(vec):
            session.add(
                PCAProjection(
                    latent_point_id=points[idx].id,
                    config_id=pca_cfg.id,
                    dim=dim,
                    value=float(val)
                )
            )

    session.commit()

    # 6) Optional plots
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

        plot_explained_variance(pca)
        plt.savefig(os.path.join(plot_dir, f"pca_{run_id}_explained_variance.png"))
        plt.close()

        plot_principal_components(pca)
        plt.savefig(os.path.join(plot_dir, f"pca_{run_id}_components.png"))
        plt.close()

    return pca_cfg.id
