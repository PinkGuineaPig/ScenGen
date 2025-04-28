# import io
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# from datetime import datetime, timezone

# from Backend.app import db
# from Backend.app.models.run_models import ModelRunConfig, ModelRun, ModelLossHistory
# from Backend.app.models.latent_models import LatentPoint, LatentInput
# from Pytorch.dataset.psql_dataset import PSQLDataset
# from Pytorch.models.lstm_vae import LSTMVAE
# from Pytorch.models.loss import vae_loss

# from torchinfo import summary

# def train_vae_for_config(app, config_id):
#     """
#     Train or retrain a single ModelRun attached to ModelRunConfig.
#     If an existing run exists, increments its version and clears old associated data.
#     Persists new ModelLossHistory, LatentPoints, and LatentInputs.

#     Returns:
#         run.id (int)
#     """
#     with app.app_context():
#         session = db.session
#         try:
#             session.begin()

#             # Load configuration
#             cfg = session.query(ModelRunConfig).get(config_id)
#             if cfg is None:
#                 raise RuntimeError(f"Config {config_id} not found.")

#             # Determine run: reuse or create
#             now = datetime.now(timezone.utc)
#             if cfg.run:
#                 run = cfg.run
#                 run.version += 1
#                 run.created_at = now
#                 # Clear previous losses and latent points
#                 session.query(ModelLossHistory).filter_by(model_run_id=run.id).delete()
#                 session.query(LatentPoint).filter_by(model_run_id=run.id).delete()
#             else:
#                 run = ModelRun(
#                     config_id  = config_id,
#                     version    = 1,
#                     created_at = now,
#                     model_blob = b''
#                 )
#                 session.add(run)
#                 session.flush()

#             # Prepare data
#             params = cfg.parameters
#             pairs  = cfg.currency_pairs
#             seq_len = params.get('seq_len', 6)
#             dataset = PSQLDataset(app, pairs, seq_len=seq_len)
#             if len(dataset) < seq_len:
#                 raise RuntimeError("Insufficient data length.")


#             # Hyperparameters
#             hidden_size = params.get('hidden_size', 64)
#             latent_dim  = params.get('latent_dim', 6)
#             batch_size  = params.get('batch_size', 32)
#             epochs      = params.get('epochs', 60)
#             lr          = params.get('learning_rate', 1e-3)
#             base_kl     = params.get('base_kl_weight', 0.1)
#             num_layers  = params.get('num_layers', 1)
#             test_every  = params.get('test_every', 1)
#             bidirectional = params.get('bidirectional', False)


#             # Split dataset
#             # Prepare dataset
#             params = cfg.parameters
#             pairs  = cfg.currency_pairs
#             seq_len = params.get('seq_len', 6)
#             dataset = PSQLDataset(app, pairs, seq_len=seq_len)
#             if len(dataset) < seq_len:
#                 raise RuntimeError("Insufficient data length.")

#             # Correct sequential split
#             n_total = len(dataset)
#             n_train = int(0.8 * n_total)  # 20% for training

#             # Create explicit subsets
#             train_ds = torch.utils.data.Subset(dataset, indices=list(range(0, n_train)))
#             test_ds  = torch.utils.data.Subset(dataset, indices=list(range(n_train, n_total)))

#             # DataLoaders
#             train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)   # Shuffle training
#             test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)  # No shuffle testing

#             # Model setup
#             sample = dataset[0]
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             model = LSTMVAE(
#                 input_size=sample.shape[1],
#                 hidden_size=hidden_size,
#                 latent_dim=latent_dim,
#                 num_layers=num_layers,
#                 bidirectional=bidirectional
#             ).to(device)

#             print(summary(model, input_size=(batch_size, seq_len, sample.shape[1])))

#             optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#             # Training loop
#             print(f"🚀 VAE training start config={config_id}, run={run.id}, version={run.version}")
#             for epoch in range(1, epochs + 1):
#                 model.train()
#                 sum_loss = sum_kl = 0.0
#                 weight = base_kl * min(1.0, epoch / epochs)
#                 #weight = 0
#                 for batch in train_loader:
#                     data = batch.to(device)
#                     recon, mu, logvar = model(data)
#                     loss = vae_loss(recon, data, mu, logvar, weight)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     sum_loss += loss.item()
#                     sum_kl   += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)).item()
#                 avg_loss = sum_loss / len(train_loader)
#                 avg_kl   = sum_kl   / len(train_loader)
#                 session.add_all([
#                     ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train',    value=avg_loss),
#                     ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train_kl', value=avg_kl)
#                 ])

#                 if epoch % test_every == 0:
#                     model.eval()
#                     test_loss = test_kl = 0.0
#                     with torch.no_grad():
#                         for batch in test_loader:
#                             data = batch.to(device)
#                             recon, mu, logvar = model(data)
#                             test_loss += nn.MSELoss()(recon, data).item()
#                             test_kl   += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)).item()
#                     session.add_all([
#                         ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test',    value=test_loss/len(test_loader)),
#                         ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test_kl', value=test_kl  /len(test_loader))
#                     ])
#                     print(f"Epoch {epoch:03d}: Train Loss = {avg_loss:.4f} | Train KL = {avg_kl:.4f}")
            
#             # Save model blob
#             buffer = io.BytesIO()
#             torch.save(model.state_dict(), buffer)
#             buffer.seek(0)
#             run.model_blob = buffer.getvalue()

#             # Save latent points and inputs
#             print("💾 Saving latent points and inputs...")
#             dates = dataset.get_common_dates()
#             for i in range(len(dataset)):
#                 x = dataset[i].unsqueeze(0).to(device)
#                 mu, logvar = model.encode(x)
#                 z = model.reparameterize(mu, logvar).squeeze(0).tolist()
#                 lp = LatentPoint(model_run_id=run.id, start_date=dates[i], lag=seq_len-1, latent_vector=z)
#                 session.add(lp)
#                 session.flush()
#                 #for pair in dataset.get_pair_codes():
#                 #    for d in dates[i:i+seq_len]:
#                 #        session.add(LatentInput(latent_point_id=lp.id, currency_pair=pair, date=d, source_table='exchange_rate'))

#             # Commit
#             session.commit()
#             print(f"✅ Completed VAE run {run.id} for config {config_id}")
#             return run.id
#         except Exception as e:
#             session.rollback()
#             print(f"❌ Training failed for config {config_id}: {e}")
#             raise




import io
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timezone

from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig, ModelRun, ModelLossHistory
from Backend.app.models.latent_models import LatentPoint
from Pytorch.dataset.psql_dataset import PSQLDataset
from Pytorch.models.lstm_vae import LSTMVAE
from Pytorch.models.loss import vae_loss
from torchinfo import summary  # <-- Add this line

logger = logging.getLogger(__name__)

import io
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timezone

from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig, ModelRun, ModelLossHistory
from Backend.app.models.latent_models import LatentPoint
from Pytorch.dataset.psql_dataset import PSQLDataset
from Pytorch.models.loss import vae_loss
from Pytorch.models.lstm_vae import LSTMVAE
from torchinfo import summary

logger = logging.getLogger(__name__)

# Use PSQLDataset directly for sliding windows

def train_vae_for_config(app, config_id, log_interval=10):
    """
    Train VAE on overlapping windows provided by PSQLDataset.
    """
    with app.app_context():
        session = db.session
        try:
            session.begin()
            # Load or create run
            cfg = session.get(ModelRunConfig, config_id)
            if cfg is None:
                raise RuntimeError(f"Config {config_id} not found")
            now = datetime.now(timezone.utc)
            run = cfg.run or ModelRun(
                config_id=config_id, version=1, created_at=now, model_blob=b''
            )
            run.version = run.version + 1 if cfg.run else 1
            run.created_at = now
            session.add(run)
            if cfg.run:
                session.query(ModelLossHistory).filter_by(model_run_id=run.id).delete()
                session.query(LatentPoint).filter_by(model_run_id=run.id).delete()
            session.flush()

            # Prepare dataset
            params = cfg.parameters
            pairs = cfg.currency_pairs
            # Support both old 'seq_len' and new 'window_size'
            seq_len = params.get('seq_len')
            window_size = params.get('window_size', seq_len)
            if window_size is None:
                raise RuntimeError("Configuration must include 'seq_len' or 'window_size'.")

            # PSQLDataset provides overlapping windows of seq_len=window_size
            full_ds = PSQLDataset(app, pairs, seq_len=window_size)
            if len(full_ds) == 0:
                raise RuntimeError(f"Need at least {window_size} samples, got {len(full_ds)}")

            # Split windows into train/test
            n = len(full_ds)
            n_train = int(0.90 * n)
            train_ds = Subset(full_ds, list(range(n_train)))
            test_ds = Subset(full_ds, list(range(n_train, n)))

            batch_size = params['batch_size']
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Model init
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sample = full_ds[0]
            model = LSTMVAE(
                input_size=sample.shape[1],
                hidden_size=params['hidden_size'],
                latent_dim=params['latent_dim'],
                num_layers=params.get('num_layers',1),
                bidirectional=params.get('bidirectional',False),
                max_seq_len=window_size
            ).to(device)
            logger.info(summary(model, input_size=(batch_size, window_size, sample.shape[1])))

            # Optimizer and hyperparams
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            epochs = params['epochs']
            base_kl = params['base_kl_weight']
            test_every = params.get('test_every',1)

            # Training loop
            for epoch in range(1, epochs + 1):
                model.train()
                sum_loss = sum_kl = 0.0
                kl_weight = base_kl * min(1.0, epoch / epochs)

                for batch_idx, batch in enumerate(train_loader, start=1):
                    data = batch.to(device)
                    optimizer.zero_grad()
                    recon, mu, logvar = model(data)
                    loss = vae_loss(recon, data, mu, logvar, kl_weight=kl_weight)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    sum_loss += loss.item()
                    sum_kl += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                               / data.size(0)).item()

                    if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
                        print(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} "
                              f"Loss: {loss.item():.4f}")

                avg_train_loss = sum_loss / len(train_loader)
                avg_train_kl = sum_kl / len(train_loader)
                tot = avg_train_loss + avg_train_kl
                session.add_all([
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train',    value=avg_train_loss),
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train_kl', value=avg_train_kl),
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train_tot', value=tot)
                ])

                # Validation
                if epoch % test_every == 0:
                    model.eval()
                    sum_test_loss = sum_test_kl = 0.0
                    with torch.no_grad():
                        for batch in test_loader:
                            data = batch.to(device)
                            recon, mu, logvar = model(data)
                            loss = vae_loss(recon, data, mu, logvar, kl_weight=kl_weight)
                            sum_test_loss += loss.item()
                            sum_test_kl += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                                            / data.size(0)).item()
                    avg_test_loss = sum_test_loss / len(test_loader)
                    avg_test_kl   = sum_test_kl   / len(test_loader)
                    avg_tot   = avg_test_loss + avg_test_kl
                    session.add_all([
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test',    value=avg_test_loss),
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test_kl', value=avg_test_kl),
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test_tot', value=avg_tot)
                    ])
                    logger.info(
                        f"Epoch {epoch:03d} | "
                        f"Train: {avg_train_loss:.4f} (KL {avg_train_kl:.4f}) | "
                        f"Test:  {avg_test_loss:.4f} (KL {avg_test_kl:.4f})"
                    )

            # Save model blob
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            run.model_blob = buffer.getvalue()

            # Save latent points for each window
            dates = full_ds.get_common_dates()
            latent_points = []
            model.eval()
            with torch.no_grad():
                for idx in range(len(full_ds)):
                    window = full_ds[idx].unsqueeze(0).to(device)
                    mu, logvar = model.encode(window)
                    z = model.reparameterize(mu, logvar).squeeze(0).tolist()
                    start_date = dates[idx]
                    latent_points.append(
                        LatentPoint(
                            model_run_id=run.id,
                            start_date=start_date,
                            lag=window_size - 1,
                            latent_vector=z
                        )
                    )
            session.add_all(latent_points)
            session.commit()
            logger.info(f"✅ Successfully completed run {run.id}")
            return run.id

        except Exception:
            session.rollback()
            logger.exception(f"❌ Training failed for config {config_id}")
            raise

