import numpy as np
from Backend.app import db
from Backend.app.models.latent_models import LatentPoint, SOMProjectionConfig, SOMProjection
from Backend.app.models.run_models import ModelRun
from Pytorch.projections.som import train_som
from datetime import datetime, timezone

def train_som_for_config(app, run_id, cfg_id):
    """
    Load existing SOMProjectionConfig, train a SOM on latent points,
    and persist projection results as SOMProjection rows.
    """
    with app.app_context():
        session = db.session
        try:
            session.begin()

            # 1) Load config
            cfg = session.query(SOMProjectionConfig).get(cfg_id)
            params = cfg.additional_params or {}
            x_dim     = cfg.x_dim
            y_dim     = cfg.y_dim
            iterations= cfg.iterations
            sigma     = params.get('sigma', 1.0)
            lr        = params.get('learning_rate', 0.5)

            # 2) Fetch latent points for this run
            points = (
                session.query(LatentPoint)
                .filter_by(model_run_id=run_id)
                .order_by(LatentPoint.id)
                .all()
            )
            if not points:
                raise RuntimeError(f"No latent points found for run_id={run_id}")

            # 3) Build data matrix (n_samples, n_features)
            Z = np.vstack([p.latent_vector for p in points])

            # 4) Train the SOM
            som = train_som(
                Z,
                x_dim=x_dim,
                y_dim=y_dim,
                sigma=sigma,
                learning_rate=lr,
                iterations=iterations
            )

            # 5) Persist projection results
            for pt in points:
                vec = pt.latent_vector
                x, y = som.winner(np.asarray(vec))
                session.add(
                    SOMProjection(
                        latent_point_id=pt.id,
                        config_id      =cfg_id,
                        x        =int(x),
                        y        =int(y),
                        created_at     =datetime.now(timezone.utc)
                    )
                )

            session.commit()
            print(f"✅ SOM training complete for config {cfg_id}")
        except:
            session.rollback()
            raise
