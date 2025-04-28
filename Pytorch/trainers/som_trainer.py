import numpy as np
from Backend.app import db
from Backend.app.models.latent_models import LatentPoint, SOMProjectionConfig, SOMProjection
from Backend.app.models.run_models import ModelRunConfig, ModelRun
from Pytorch.projections.som import train_som
from datetime import datetime, timezone


def train_som_for_config(app, model_cfg_id, som_cfg_id):
    """
    Load existing SOMProjectionConfig, train a SOM on latent points,
    and persist projection results as SOMProjection rows.

    :param app: Flask application (for app context)
    :param model_cfg_id: ID of the ModelRunConfig
    :param som_cfg_id:   ID of the SOMProjectionConfig to run
    """
    with app.app_context():
        session = db.session

        # 1) Load and validate the ModelRunConfig
        config = session.query(ModelRunConfig).get(model_cfg_id)
        if not config:
            raise ValueError(f"ModelRunConfig {model_cfg_id} not found")

        # 2) Find the associated ModelRun for this config
        run = session.query(ModelRun).filter_by(config_id=model_cfg_id).first()
        if not run:
            raise ValueError(f"No ModelRun found for ModelRunConfig {model_cfg_id}")
        run_id = run.id

        # 3) Load and validate SOMProjectionConfig
        cfg = session.query(SOMProjectionConfig).get(som_cfg_id)
        if not cfg or cfg.model_config_id != model_cfg_id:
            raise ValueError(f"SOMProjectionConfig {som_cfg_id} not valid for model_cfg {model_cfg_id}")

        # 4) Fetch latent points for this run
        points = (
            session.query(LatentPoint)
                   .filter_by(model_run_id=run_id)
                   .order_by(LatentPoint.id)
                   .all()
        )
        if not points:
            raise RuntimeError(f"No latent points found for run_id={run_id}")

        # 5) Build data matrix
        Z = np.vstack([p.latent_vector for p in points])

        # 6) Unpack hyperparameters
        params = cfg.additional_params or {}
        sigma = params.get('sigma', 1.0)
        lr    = params.get('learning_rate', 0.5)
        x_dim = cfg.x_dim
        y_dim = cfg.y_dim
        iterations = cfg.iterations

        # 7) Train the SOM
        som = train_som(
            Z,
            x_dim=x_dim,
            y_dim=y_dim,
            sigma=sigma,
            learning_rate=lr,
            iterations=iterations
        )

        # 8) Persist projection results in bulk
        projections = []
        for pt in points:
            x, y = som.winner(np.asarray(pt.latent_vector))
            projections.append(
                SOMProjection(
                    latent_point_id=pt.id,
                    config_id=som_cfg_id,
                    x=int(x),
                    y=int(y),
                    created_at=datetime.now(timezone.utc)
                )
            )
        session.bulk_save_objects(projections)
        session.commit()

    return som_cfg_id
