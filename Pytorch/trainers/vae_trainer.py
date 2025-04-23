import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime, timezone

from Backend.app import db
from Backend.app.models.run_models import ModelRun, ModelRunConfig, ModelLossHistory
from Backend.app.models.latent_models import LatentPoint, LatentInput
from Pytorch.dataset.psql_dataset import PSQLDataset
from Pytorch.models.lstm_vae import LSTMVAE
from Pytorch.models.loss import vae_loss


def train_vae_for_config(app, currency_pairs, hyperparams):
    """
    Train a LSTM-VAE on specified currency_pairs according to hyperparams,
    store ModelRunConfig, ModelRun, ModelLossHistory, LatentPoint, and LatentInput.

    :param app: Flask app
    :param currency_pairs: list of currency strings, e.g. ['EUR/USD']
    :param hyperparams: dict with keys:
        'seq_len', 'hidden_size', 'latent_dim', 'batch_size', 'epochs',
        'learning_rate', 'base_kl_weight', 'test_every'
    :returns: created ModelRun id
    """
    with app.app_context():
        session = db.session
        try:
            session.begin()

            # Load and validate dataset
            seq_len = hyperparams.get('seq_len', 6)
            dataset = PSQLDataset(app, currency_pairs, seq_len=seq_len)
            if len(dataset) < seq_len:
                raise RuntimeError('Not enough data for training.')

            # Split dataset
            n = len(dataset)
            train_n = int(0.7 * n)
            val_n = int(0.15 * n)
            test_n = n - train_n - val_n
            train_ds, _, test_ds = random_split(dataset, [train_n, val_n, test_n])

            # Hyperparams
            hidden_size = hyperparams.get('hidden_size', 64)
            latent_dim = hyperparams.get('latent_dim', 6)
            batch_size = hyperparams.get('batch_size', 32)
            epochs = hyperparams.get('epochs', 60)
            lr = hyperparams.get('learning_rate', 1e-3)
            base_kl = hyperparams.get('base_kl_weight', 0.1)
            test_every = hyperparams.get('test_every', 1)

            # DataLoaders
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Model and optimizer
            sample = dataset[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMVAE(sample.shape[1], hidden_size, latent_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Create config
            config = ModelRunConfig(
                model_type='LSTMVAE',
                currency_pairs=currency_pairs,
                parameters={
                    'seq_len': seq_len,
                    'hidden_size': hidden_size,
                    'latent_dim': latent_dim,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'learning_rate': lr,
                    'base_kl_weight': base_kl
                },
                created_at=datetime.now(timezone.utc)
            )
            session.add(config)
            session.flush()

            # Create run
            last_ver = session.query(db.func.max(ModelRun.version)).filter_by(config_id=config.id).scalar() or 0
            run = ModelRun(
                config_id=config.id,
                version=last_ver + 1,
                created_at=datetime.now(timezone.utc),
                model_blob=b''
            )
            session.add(run)
            session.flush()

            # Training loop
            print('Starting VAE training...')
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = train_kl = 0.0
                weight = base_kl * min(1.0, epoch / epochs)

                for batch in train_loader:
                    data = batch.to(device)
                    recon, mu, logvar = model(data)
                    loss = vae_loss(recon, data, mu, logvar, weight)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_kl += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)).item()

                avg_loss = train_loss / len(train_loader)
                avg_kl = train_kl / len(train_loader)
                session.add_all([
                    ModelLossHistory(run=run, epoch=epoch, loss_type='train', value=avg_loss),
                    ModelLossHistory(run=run, epoch=epoch, loss_type='train_kl', value=avg_kl)
                ])
                print(f'Epoch {epoch}/{epochs} | Loss={avg_loss:.4f} | KL={avg_kl:.4f}')

                if epoch % test_every == 0:
                    model.eval()
                    val_loss = val_kl = 0.0
                    with torch.no_grad():
                        for batch in test_loader:
                            data = batch.to(device)
                            recon, mu, logvar = model(data)
                            val_loss += nn.MSELoss()(recon, data).item()
                            val_kl += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)).item()
                    session.add_all([
                        ModelLossHistory(run=run, epoch=epoch, loss_type='test', value=val_loss / len(test_loader)),
                        ModelLossHistory(run=run, epoch=epoch, loss_type='test_kl', value=val_kl / len(test_loader))
                    ])
                    print(f'Test @ {epoch} | Loss={val_loss/len(test_loader):.4f} | KL={val_kl/len(test_loader):.4f}')

            # Save model blob
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            run.model_blob = buffer.getvalue()

            # Persist latent points
            print('Saving latent points...')
            dates = dataset.get_common_dates()
            for i in range(len(dataset)):
                x = dataset[i].unsqueeze(0).to(device)
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar).squeeze(0).tolist()
                lp = LatentPoint(model_run_id=run.id, start_date=dates[i], lag=seq_len - 1, latent_vector=z)
                session.add(lp)
                session.flush()
                
                for pair in dataset.get_pair_codes():              # loop through all pairs
                    for d in dates[i:i + seq_len]:                 # loop through each date in the window
                        session.add(LatentInput(
                            latent_point_id=lp.id,
                            currency_pair=pair,
                            date=d,
                            source_table='exchange_rate'
                        ))

            session.commit()
            print('VAE training complete, run_id:', run.id)
            return run.id

        except Exception:
            session.rollback()
            raise
