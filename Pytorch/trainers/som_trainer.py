import os
import numpy as np
import matplotlib.pyplot as plt
from Backend.app import db
from Backend.app.models.latent_models import LatentPoint, SOMProjectionConfig, SOMProjection
from Pytorch.projections.som import (
    train_som,
    plot_u_matrix,
    plot_data_hits,
    plot_component_planes
)


def run_som_for_run(session, run_id, hyperparams, plot_dir=None):
    """
    Train a Self-Organizing Map on latent vectors for a given model run,
    persist the results in the database, and optionally save diagnostic plots.

    :param session: SQLAlchemy session
    :param run_id: ID of the ModelRun to process
    :param hyperparams: dict containing SOM hyperparameters:
        - som_x_dim (int)
        - som_y_dim (int)
        - som_iterations (int)
        - som_sigma (float)
        - som_learning_rate (float)
    :param plot_dir: optional path to save plots; if None, no files are written
    :returns: id of created SOMProjectionConfig
    """
    # 1) Fetch latent points for this run
    points = (
        session.query(LatentPoint)
        .filter_by(model_run_id=run_id)
        .order_by(LatentPoint.id)
        .all()
    )
    if not points:
        raise ValueError(f"No LatentPoint entries found for run_id={run_id}")

    # 2) Build data matrix of shape (n_samples, n_features)
    Z = np.vstack([p.latent_vector for p in points])

    # 3) Read SOM hyperparameters (with defaults)
    x_dim      = hyperparams.get('som_x_dim', 10)
    y_dim      = hyperparams.get('som_y_dim', 10)
    iterations = hyperparams.get('som_iterations', 1000)
    sigma      = hyperparams.get('som_sigma', 1.0)
    lr         = hyperparams.get('som_learning_rate', 0.5)

    # 4) Train the SOM
    som = train_som(
        Z,
        x_dim=x_dim,
        y_dim=y_dim,
        sigma=sigma,
        learning_rate=lr,
        iterations=iterations
    )

    # 5) Persist SOMProjectionConfig
    som_cfg = SOMProjectionConfig(
        model_run_id=run_id,
        x_dim=x_dim,
        y_dim=y_dim,
        iterations=iterations,
        additional_params={'sigma': sigma, 'learning_rate': lr}
    )
    session.add(som_cfg)
    session.flush()  # populates som_cfg.id

    # 6) Persist SOMProjection rows
    for pt, vec in zip(points, Z):
        x, y = som.winner(vec)
        session.add(
            SOMProjection(
                latent_point_id=pt.id,
                config_id=som_cfg.id,
                x=int(x),
                y=int(y)
            )
        )

    # 7) Commit the transaction so projections are saved
    session.commit()

    # 8) Optionally generate and save plots
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

        # U-Matrix
        plot_u_matrix(som)
        plt.savefig(os.path.join(plot_dir, f"som_{run_id}_u_matrix.png"))
        plt.close()

        # Data hits (unlabeled)
        plot_data_hits(som, Z)
        plt.savefig(os.path.join(plot_dir, f"som_{run_id}_hits.png"))
        plt.close()

        # Component planes
        plot_component_planes(som)
        plt.savefig(os.path.join(plot_dir, f"som_{run_id}_component_planes.png"))
        plt.close()

    return som_cfg.id
