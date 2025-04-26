# Pytorch/trainers/pca_trainer.py

import numpy as np
from Backend.app import db
from Backend.app.models.latent_models import LatentPoint, PCAProjectionConfig, PCAProjection
from Pytorch.projections.pca import fit_pca

def run_pca_for_run(session, run_id, hyperparams):
    """
    Train PCA on latent vectors for a given model run,
    persist the config and all component projections.

    :param session: SQLAlchemy session
    :param run_id:   ID of the ModelRun to process
    :param hyperparams: dict containing PCA hyperparameters:
                         - n_components (int)
                         - additional_params (dict)
    :returns: ID of the newly created PCAProjectionConfig
    """

    # 1) Fetch latent points
    points = (session.query(LatentPoint)
                     .filter_by(model_run_id=run_id)
                     .order_by(LatentPoint.id)
                     .all())
    if not points:
        raise ValueError(f"No LatentPoint entries found for run_id={run_id}")

    # 2) Build data matrix [n_samples, n_features]
    Z = np.vstack([p.latent_vector for p in points])

    # 3) Unpack PCA hyperparameters
    n_components = hyperparams.get('n_components', min(Z.shape))
    additional   = hyperparams.get('additional_params', {})

    # 4) Fit PCA
    pca, X_reduced = fit_pca(Z, n_components, additional)

    # 5) Persist a new PCAProjectionConfig
    pca_cfg = PCAProjectionConfig(
        model_run_id      = run_id,
        config_id         = session.query(db.func.max(PCAProjectionConfig.config_id)).scalar() or run_id,
        n_components      = n_components,
        additional_params = additional
    )
    session.add(pca_cfg)
    session.flush()  # now pca_cfg.id is available

    # 6) Bulk‐insert every projection row
    projections = []
    for idx, vec in enumerate(X_reduced):
        for dim, val in enumerate(vec):
            projections.append(
                PCAProjection(
                    latent_point_id = points[idx].id,
                    config_id       = pca_cfg.id,
                    dim             = dim,
                    value           = float(val)
                )
            )

    session.add_all(projections)
    session.commit()

    return pca_cfg.id
