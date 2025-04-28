import numpy as np
from Backend.app import db
from Backend.app.models.latent_models import LatentPoint, PCAProjectionConfig, PCAProjection
from Backend.app.models.run_models import ModelRunConfig, ModelRun
from Pytorch.projections.pca import fit_pca


def run_pca_for_config(app, model_cfg_id, pca_cfg_id):
    """
    Background PCA trainer that loads parameters from the database via relationships,
    fits PCA, stores both the projections and the explained_variance_ratio into the config.
    """
    with app.app_context():
        session = db.session

        # 1) Load the model config and its PCA sub-config
        mcfg = session.get(ModelRunConfig, model_cfg_id)
        if not mcfg:
            raise ValueError(f"ModelRunConfig {model_cfg_id} not found")
        pca_cfg = next((c for c in mcfg.pca_configs if c.id == pca_cfg_id), None)
        if not pca_cfg:
            raise ValueError(f"PCAProjectionConfig {pca_cfg_id} not valid for model_cfg {model_cfg_id}")

        # 2) Grab the one-to-one run and its latent points
        run = mcfg.run
        if not run or not run.latent_points:
            raise RuntimeError(f"No ModelRun/LatentPoint for config {model_cfg_id}")
        points = sorted(run.latent_points, key=lambda p: p.id)

        # 3) Build data matrix
        Z = np.vstack([pt.latent_vector for pt in points])

        # 4) Unpack & normalize hyperparams
        params = dict(pca_cfg.additional_params or {})
        if 'solver' in params:
            # scikit-learn expects 'svd_solver'
            params['svd_solver'] = params.pop('solver')
        n_comp = pca_cfg.n_components

        # 5) Fit PCA
        pca_obj, X_reduced = fit_pca(Z, n_comp, params)

        # 6) Persist explained variance ratio on the config
        pca_cfg.explained_variance = pca_obj.explained_variance_ratio_.tolist()
        pca_cfg.components         = pca_obj.components_.tolist()
        session.add(pca_cfg)
        session.flush()

        # 7) Bulk-create projection rows
        projections = []
        for pt, vec in zip(points, X_reduced):
            for dim, val in enumerate(vec):
                projections.append(PCAProjection(
                    latent_point_id=pt.id,
                    config_id      =pca_cfg_id,
                    dim            =dim,
                    value          =float(val)
                ))
        session.bulk_save_objects(projections)
        session.commit()

    return pca_cfg_id