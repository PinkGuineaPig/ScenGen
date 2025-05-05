# File: C:\Users\Kevin\Documents\ScenGen\Pytorch\tests\cluster_analysis.py
#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from minisom import MiniSom
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import io
import logging
from datetime import datetime
from sqlalchemy import and_

from Backend.app import db, create_app
from Backend.app.models.latent_models import LatentPoint
from Backend.app.models.run_models import ModelRun, ModelRunConfig
from Pytorch.dataset.psql_dataset import InMemoryWindowDataset
from Pytorch.models.lstm_vae import LSTMVAE

LOG_DIR = os.getenv('LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'cluster_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def plot_pca(latent_vectors, output_dir, run_id):
    pca = PCA(n_components=3)
    latent_3d = pca.fit_transform(latent_vectors)
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance: PC1={explained_variance[0]:.2%}, "
                f"PC2={explained_variance[1]:.2%}, PC3={explained_variance[2]:.2%}")

    plt.figure(figsize=(10, 8))
    plt.scatter(latent_3d[:, 0], latent_3d[:, 1], alpha=0.5, s=10)
    plt.title(f'2D PCA of Test Set Latent Space (Run {run_id})\n'
              f'Explained Variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    output_2d_path = os.path.join(output_dir, f'latent_pca_2d_{run_id}.png')
    plt.savefig(output_2d_path)
    plt.close()
    logger.info(f"Saved 2D PCA plot to {output_2d_path}")

    angles = [0, 45, 90]
    for angle in angles:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], alpha=0.5, s=10)
        ax.set_title(f'3D PCA of Test Set Latent Space (Run {run_id})\n'
                     f'Explained Variance: PC1={explained_variance[0]:.2%}, '
                     f'PC2={explained_variance[1]:.2%}, PC3={explained_variance[2]:.2%}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.view_init(elev=30, azim=angle)
        ax.grid(True)
        output_3d_path = os.path.join(output_dir, f'latent_pca_3d_angle_{angle}_{run_id}.png')
        plt.savefig(output_3d_path)
        plt.close()
        logger.info(f"Saved 3D PCA plot (angle={angle}) to {output_3d_path}")

def plot_som_node_reconstructions(model, dataset, device, latent_vectors, dates, dataset_dates, node_idx,
                                  closest_indices, output_path, ma_price_min, ma_price_max,
                                  squared_return_min, squared_return_max):
    model.eval()
    num_samples = len(closest_indices)
    plt.figure(figsize=(15, num_samples * 6))

    with torch.no_grad():
        for i, idx in enumerate(closest_indices):
            window, ranges = dataset[idx]
            window = window.unsqueeze(0).to(device)
            ma_price_min_i, ma_price_max_i, squared_return_min_i, squared_return_max_i = ranges
            ma_price_min_i = ma_price_min_i.item()
            ma_price_max_i = ma_price_max_i.item()
            squared_return_min_i = squared_return_min_i.item()
            squared_return_max_i = squared_return_max_i.item()
            recon, _, _ = model(window)
            window = window.cpu().numpy()[0]
            recon = recon.cpu().numpy()[0]

            window_start_idx = idx
            window_end_idx = idx + len(window)
            window_timestamps = dataset_dates[window_start_idx:window_end_idx]
            logger.info(f"Node {node_idx} Sample {i+1} timestamps: {window_timestamps}")

            x_ma_price = window[:, 0] * (ma_price_max_i - ma_price_min_i) + ma_price_min_i
            recon_ma_price = recon[:, 0] * (ma_price_max_i - ma_price_min_i) + ma_price_min_i
            x_squared_return = window[:, 1] * (squared_return_max_i - squared_return_min_i) + squared_return_min_i
            recon_squared_return = recon[:, 1] * (squared_return_max_i - squared_return_min_i) + squared_return_min_i

            plt.subplot(num_samples, 2, 2 * i + 1)
            plt.plot(x_ma_price, label='Input MA Price')
            plt.plot(recon_ma_price, label='Recon MA Price', linestyle='--')
            plt.title(f'Node {node_idx} Sample {i+1} MA Price (Start: {dates[idx]})')
            plt.legend()

            plt.subplot(num_samples, 2, 2 * i + 2)
            plt.plot(x_squared_return, label='Input Squared Return')
            plt.plot(recon_squared_return, label='Recon Squared Return', linestyle='--')
            plt.title(f'Node {node_idx} Sample {i+1} Squared Return')
            plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved reconstructions for SOM node {node_idx} to {output_path}")

def compile_pngs_to_pdf(png_files, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    for png_path in png_files:
        img = Image.open(png_path)
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        scaled_width = width * 0.9
        scaled_height = scaled_width * aspect
        c.drawImage(png_path, (width - scaled_width) / 2, (height - scaled_height) / 2,
                    width=scaled_width, height=scaled_height)
        c.showPage()
    c.save()
    logger.info(f"Compiled {len(png_files)} plots into {output_pdf}")

def analyze_clusters(app, run_id, som_grid=(6, 6), sample_limit=None, output_dir='cluster_analysis'):
    logger.info(f"Starting cluster analysis for run {run_id} (test set)")
    os.makedirs(output_dir, exist_ok=True)

    with app.app_context():
        session = db.session
        run = session.get(ModelRun, run_id)
        if not run:
            raise RuntimeError(f"Run {run_id} not found")
        cfg = session.get(ModelRunConfig, run.config_id)
        p = cfg.parameters
        seq_len = p['seq_len']
        hidden_size = p['hidden_size']
        latent_dim = p['latent_dim']
        num_layers = p.get('num_layers', 1)
        bidirectional = p.get('bidirectional', False)

        ds = InMemoryWindowDataset(app, cfg.currency_pairs, seq_len=seq_len,
                                   stride=p.get('stride', 1), ffill_limit=p.get('ffill_limit', 2))
        dataset_dates = ds.get_common_dates()
        if len(dataset_dates) < seq_len:
            raise RuntimeError(f"Dataset too small: {len(dataset_dates)} dates, need at least {seq_len}")

        N = len(ds)
        n_train = int(0.8 * N)
        test_indices = range(n_train, N)
        test_dates = [dataset_dates[i + seq_len - 1] for i in test_indices]
        test_dates = [pd.Timestamp(d).tz_localize(None) for d in test_dates]
        logger.info(f"Test set: {len(test_dates)} dates, from {min(test_dates)} to {max(test_dates)}")

        latent_points = session.query(LatentPoint).filter(
            and_(
                LatentPoint.model_run_id == run_id,
                LatentPoint.start_date.in_(test_dates)
            )
        ).all()
        if not latent_points:
            raise RuntimeError(f"No latent points found for run {run_id} in test set")
        latent_vectors = np.array([lp.latent_vector for lp in latent_points])
        dates = [lp.start_date for lp in latent_points]
        logger.info(f"Loaded {len(latent_vectors)} test set latent vectors of dim {latent_dim}")

        if sample_limit and len(latent_vectors) > sample_limit:
            indices = np.random.choice(len(latent_vectors), sample_limit, replace=False)
            latent_vectors = latent_vectors[indices]
            dates = [dates[i] for i in indices]
            logger.info(f"Sampled {sample_limit} test set latent vectors")
        else:
            logger.info(f"Using all {len(latent_vectors)} test set latent vectors (no sampling)")

        plot_pca(latent_vectors, output_dir, run_id)

        som = MiniSom(som_grid[0], som_grid[1], latent_dim, sigma=1.0, learning_rate=0.5)
        som.random_weights_init(latent_vectors)
        som.train_random(latent_vectors, 1000)
        logger.info(f"Trained SOM {som_grid[0]}x{som_grid[1]}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMVAE(
            input_size=ds.input_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        ).to(device)
        buf = io.BytesIO(run.model_blob)
        state = torch.load(buf, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info(f"Loaded model for run {run_id}")

        dataset_dates = dataset_dates[seq_len-1:]
        if len(dataset_dates) < len(dates):
            raise RuntimeError(f"Dataset dates ({len(dataset_dates)}) fewer than latent points ({len(dates)})")
        date_to_idx = {pd.Timestamp(d).tz_localize(None): i for i, d in enumerate(dataset_dates)}

        png_files = []
        for i in range(som_grid[0]):
            for j in range(som_grid[1]):
                node_idx = f'{i},{j}'
                node_weights = som.get_weights()[i, j]
                distances = np.linalg.norm(latent_vectors - node_weights, axis=1)
                closest_indices = np.argsort(distances)[:3]
                closest_dates = [dates[idx] for idx in closest_indices]

                try:
                    dataset_indices = [date_to_idx[pd.Timestamp(d).tz_localize(None)] for d in closest_dates]
                except KeyError as e:
                    logger.error(f"Date not found in dataset: {e}")
                    continue

                output_path = os.path.join(output_dir, f'som_node_{i}_{j}_{run_id}.png')
                plot_som_node_reconstructions(
                    model, ds, device, latent_vectors, dates, dataset_dates, node_idx,
                    closest_indices, output_path,
                    ds.ma_price_min, ds.ma_price_max,
                    ds.squared_return_min, ds.squared_return_max
                )
                png_files.append(output_path)

        pdf_path = os.path.join(output_dir, f'som_reconstructions_{run_id}.pdf')
        compile_pngs_to_pdf(png_files, pdf_path)

        logger.info(f"Completed cluster analysis for run {run_id}")

if __name__ == '__main__':
    DEV_DB = "postgresql://scengen_dev_user:dev_secret@localhost/scengen_dev"
    app = create_app({"SQLALCHEMY_DATABASE_URI": DEV_DB})
    analyze_clusters(app, run_id=10, som_grid=(6, 6), sample_limit=None)