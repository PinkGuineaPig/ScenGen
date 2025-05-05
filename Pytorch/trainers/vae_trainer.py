import io
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CyclicLR
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

from Backend.app import db, create_app
from Backend.app.models.run_models import ModelRunConfig, ModelRun, ModelLossHistory
from Backend.app.models.latent_models import LatentPoint
from Pytorch.dataset.psql_dataset import InMemoryWindowDataset
from Pytorch.models.lstm_vae import LSTMVAE

LOG_DIR = os.getenv('LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'vae_trainer.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def plot_reconstructions(model, loader, device, num_samples=5, epoch=0, log_dir=LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        x, ranges = batch
        x = x.to(device)[:num_samples]
        x_smoothed = model.ema(x)  # Only squared returns are smoothed
        ma_price_min, ma_price_max, ret_min, ret_max = [r[:num_samples] for r in ranges]
        recon, _, _ = model(x)
        x = x.cpu().numpy()
        x_smoothed = x_smoothed.cpu().numpy()
        recon = recon.cpu().numpy()
        ma_price_min = ma_price_min.cpu().numpy()
        ma_price_max = ma_price_max.cpu().numpy()
        ret_min = ret_min.cpu().numpy()
        ret_max = ret_max.cpu().numpy()

        plt.figure(figsize=(15, num_samples * 6))
        for i in range(num_samples):
            x_ma_price = x[i, :, 0] * (ma_price_max[i] - ma_price_min[i]) + ma_price_min[i]
            recon_ma_price = recon[i, :, 0] * (ma_price_max[i] - ma_price_min[i]) + ma_price_min[i]
            x_ret = x[i, :, 1] * (ret_max[i] - ret_min[i]) + ret_min[i]
            x_smoothed_ret = x_smoothed[i, :, 1] * (ret_max[i] - ret_min[i]) + ret_min[i]
            recon_ret = recon[i, :, 1] * (ret_max[i] - ret_min[i]) + ret_min[i]

            plt.subplot(num_samples, 2, 2 * i + 1)
            plt.plot(x_ma_price, label='Input MA Price')
            plt.plot(recon_ma_price, label='Recon MA Price', linestyle='--')
            plt.title(f'Sample {i+1} MA Price')
            plt.legend()
            plt.subplot(num_samples, 2, 2 * i + 2)
            plt.plot(x_ret, label='Input Return')
            plt.plot(x_smoothed_ret, label='Smoothed Return', linestyle='-.')
            plt.plot(recon_ret, label='Recon Return', linestyle='--')
            plt.title(f'Sample {i+1} Return')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'recon_ep{epoch}.png'))
        plt.close()

def train_vae_for_config(app, config_id, log_interval=100):
    logger.info(f"Starting VAE training for config {config_id}")
    with app.app_context():
        session = db.session
        try:
            session.begin()

            cfg = session.get(ModelRunConfig, config_id)
            if cfg is None:
                raise RuntimeError(f"Config {config_id} not found")
            now = datetime.now(timezone.utc)
            run = cfg.run or ModelRun(config_id=config_id, version=1, created_at=now, model_blob=b'')
            if cfg.run:
                run.version += 1
                session.query(ModelLossHistory).filter_by(model_run_id=run.id).delete()
                session.query(LatentPoint).filter_by(model_run_id=run.id).delete()
            else:
                session.add(run)
                session.flush()
            run.created_at = now

            p = cfg.parameters
            seq_len = p['seq_len']
            stride = p.get('stride', 1)
            ffill = p.get('ffill_limit', 2)
            batch_size = p['batch_size']
            lr = p['learning_rate']
            epochs = p['epochs']
            base_kl = p.get('base_kl_weight', 0.05)
            test_every = p.get('test_every', 1)
            hidden_size = p['hidden_size']
            latent_dim = p['latent_dim']
            num_layers = p.get('num_layers', 1)
            bidirectional = p.get('bidirectional', False)

            logger.info(f"Config: seq_len={seq_len}, batch_size={batch_size}, hidden_size={hidden_size}, "
                        f"latent_dim={latent_dim}, base_kl={base_kl}, lr={lr}, epochs={epochs}")

            ds = InMemoryWindowDataset(app, cfg.currency_pairs, seq_len=seq_len,
                                       stride=stride, ffill_limit=ffill)
            N = len(ds)
            if N == 0:
                raise RuntimeError(f"Need at least {seq_len} windows, got 0")
            logger.info(f"Dataset Size: {N}, Input Shape: {ds[0][0].shape}")
            n_train = int(0.8 * N)
            train_ds = Subset(ds, range(n_train))
            test_ds = Subset(ds, range(n_train, N))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=(device.type=='cuda'), drop_last=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=0, pin_memory=(device.type=='cuda'))

            train_sample, _ = next(iter(train_loader))
            train_sample = train_sample.to(device)
            logger.info(f"Train Input Min: {train_sample.min():.4f}, Max: {train_sample.max():.4f}, "
                        f"Mean: {train_sample.mean():.4f}, Std: {train_sample.std():.4f}, Shape: {train_sample.shape}")

            model = LSTMVAE(input_size=ds.input_size, hidden_size=hidden_size,
                            latent_dim=latent_dim, num_layers=num_layers,
                            bidirectional=bidirectional).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

            model_summary = summary(model, input_size=(batch_size, seq_len, ds.input_size), device=device)
            logger.info(f"Model Summary:\n{model_summary}")
            print(f"Model Summary:\n{model_summary}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=5e-4, step_size_up=500,
                                mode='triangular', cycle_momentum=False)

            best_test_recon = float('inf')
            patience = 1
            patience_counter = 0

            for epoch in range(1, epochs + 1):
                model.train()
                sum_loss = sum_kl = sum_recon = 0.0
                kl_w = base_kl * min(1.0, max(0.0, (epoch - 2) / 5.0))
                logger.info(f"Epoch {epoch}/{epochs} KL weight: {kl_w:.4f}")

                for i, (x, _) in enumerate(train_loader, 1):
                    x = x.to(device)
                    optimizer.zero_grad()
                    recon, mu, logvar = model(x)
                    recon_loss = nn.MSELoss()(recon, x)
                    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                    loss = recon_loss + kl_w * kl_div

                    if torch.isnan(loss) or torch.isinf(loss):
                        fn = os.path.join(LOG_DIR, f'nan_cfg{config_id}_ep{epoch}_batch{i}.pt')
                        torch.save(model.state_dict(), fn)
                        logger.error(f"NaN/Inf loss at ep{i}, saved to {fn}")
                        raise RuntimeError(f"NaN/Inf loss at epoch {epoch} batch {i}")

                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    scheduler.step()

                    for name, param in model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            logger.error(f"NaN/Inf in {name} at epoch {epoch} batch {i}")
                            raise RuntimeError(f"NaN/Inf in {name}")

                    sum_loss += loss.item()
                    sum_recon += recon_loss.item()
                    sum_kl += kl_div.item()

                    if i % log_interval == 0:
                        avg_loss = sum_loss / i
                        avg_recon = sum_recon / i
                        avg_kl = sum_kl / i
                        current_lr = optimizer.param_groups[0]['lr']
                        msg = (f"[Epoch {epoch} Batch {i}/{len(train_loader)}] "
                               f"Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}, "
                               f"GradNorm: {grad_norm:.4f}, LR: {current_lr:.6f}")
                        print(msg)
                        logger.info(msg)

                avg_loss = sum_loss / len(train_loader)
                avg_recon = sum_recon / len(train_loader)
                avg_kl = sum_kl / len(train_loader)
                msg = (f"[Epoch {epoch} Final] "
                       f"Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}")
                print(msg)
                logger.info(msg)

                if epoch % test_every == 0:
                    model.eval()
                    sum_te = sum_tk = sum_tr = 0.0
                    with torch.no_grad():
                        for x, _ in test_loader:
                            x = x.to(device)
                            recon, mu, logvar = model(x)
                            rl = nn.MSELoss()(recon, x)
                            kd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                            sum_te += (rl + kl_w * kd).item()
                            sum_tk += kd.item()
                            sum_tr += rl.item()

                    test_loss = sum_te / len(test_loader)
                    test_kl = sum_tk / len(test_loader)
                    test_recon = sum_tr / len(test_loader)
                    session.bulk_save_objects([
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test', value=test_loss),
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test_kl', value=test_kl),
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test_recon', value=test_recon),
                    ])
                    msg = (f"Epoch {epoch}/{epochs} Test Loss: {test_loss:.4f}, "
                           f"Test KL: {test_kl:.4f}, Test Recon: {test_recon:.4f}")
                    print(msg)
                    logger.info(msg)
                    plot_reconstructions(model, test_loader, device, epoch=epoch)

                    if test_recon < best_test_recon:
                        best_test_recon = test_recon
                        patience_counter = 0
                        buf = io.BytesIO()
                        torch.save(model.state_dict(), buf)
                        buf.seek(0)
                        run.model_blob = buf.getvalue()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break

                session.bulk_save_objects([
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train', value=sum_loss/len(train_loader)),
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train_kl', value=sum_kl/len(train_loader)),
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train_recon', value=sum_recon/len(train_loader)),
                ])
                session.commit()

            logger.info(f"Training completed run {run.id}")
            return run.id

        except Exception:
            session.rollback()
            logger.exception("Training failed for config %s", config_id)
            raise

def generate_and_bulk_save_latents(app, run_id, batch_size=64, log_interval=100):
    with app.app_context():
        session = db.session
        run = session.get(ModelRun, run_id)
        cfg = session.get(ModelRunConfig, run.config_id)
        seq_len = cfg.parameters.get('seq_len')
        window = seq_len
        stride = cfg.parameters.get('stride', 1)
        ffill_limit = cfg.parameters.get('ffill_limit', 2)

        ds = InMemoryWindowDataset(app, cfg.currency_pairs, seq_len=window, stride=stride, ffill_limit=ffill_limit)
        dates = ds.get_common_dates()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMVAE(
            input_size=ds.input_size,
            hidden_size=cfg.parameters['hidden_size'],
            latent_dim=cfg.parameters['latent_dim'],
            num_layers=cfg.parameters.get('num_layers', 1),
            bidirectional=cfg.parameters.get('bidirectional', False)
        ).to(device)
        buf = io.BytesIO(run.model_blob)
        state = torch.load(buf, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        latents = []
        idx_offset = 0
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(loader, start=1):
                x = x.to(device)
                mu, logvar = model.encode(x)
                zs = model.reparameterize(mu, logvar).cpu().tolist()
                for i, z_vec in enumerate(zs):
                    ts = dates[idx_offset + i + (window - 1)]
                    latents.append(
                        LatentPoint(model_run_id=run_id, start_date=ts, lag=window - 1, latent_vector=z_vec)
                    )
                idx_offset += len(zs)
                if batch_idx % log_interval == 0:
                    logger.info(f"Processed {idx_offset} latent points for run {run_id}")

        session.bulk_save_objects(latents)
        session.commit()
        logger.info(f"Bulk-saved {len(latents)} latent points for run {run_id}")

if __name__ == '__main__':
    DEV_DB = "postgresql://scengen_dev_user:dev_secret@localhost/scengen_dev"
    app = create_app({"SQLALCHEMY_DATABASE_URI": DEV_DB})
    train_vae_for_config(app, config_id=1)