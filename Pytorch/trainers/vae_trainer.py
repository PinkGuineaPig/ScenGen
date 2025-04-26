import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime, timezone

from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig, ModelRun, ModelLossHistory
from Backend.app.models.latent_models import LatentPoint, LatentInput
from Pytorch.dataset.psql_dataset import PSQLDataset
from Pytorch.models.lstm_vae import LSTMVAE
from Pytorch.models.loss import vae_loss

from torchinfo import summary

def train_vae_for_config(app, config_id):
    """
    Train or retrain a single ModelRun attached to ModelRunConfig.
    If an existing run exists, increments its version and clears old associated data.
    Persists new ModelLossHistory, LatentPoints, and LatentInputs.

    Returns:
        run.id (int)
    """
    with app.app_context():
        session = db.session
        try:
            session.begin()

            # Load configuration
            cfg = session.query(ModelRunConfig).get(config_id)
            if cfg is None:
                raise RuntimeError(f"Config {config_id} not found.")

            # Determine run: reuse or create
            now = datetime.now(timezone.utc)
            if cfg.run:
                run = cfg.run
                run.version += 1
                run.created_at = now
                # Clear previous losses and latent points
                session.query(ModelLossHistory).filter_by(model_run_id=run.id).delete()
                session.query(LatentPoint).filter_by(model_run_id=run.id).delete()
            else:
                run = ModelRun(
                    config_id  = config_id,
                    version    = 1,
                    created_at = now,
                    model_blob = b''
                )
                session.add(run)
                session.flush()

            # Prepare data
            params = cfg.parameters
            pairs  = cfg.currency_pairs
            seq_len = params.get('seq_len', 6)
            dataset = PSQLDataset(app, pairs, seq_len=seq_len)
            if len(dataset) < seq_len:
                raise RuntimeError("Insufficient data length.")


            # Hyperparameters
            hidden_size = params.get('hidden_size', 64)
            latent_dim  = params.get('latent_dim', 6)
            batch_size  = params.get('batch_size', 32)
            epochs      = params.get('epochs', 60)
            lr          = params.get('learning_rate', 1e-3)
            base_kl     = params.get('base_kl_weight', 0.1)
            num_layers  = params.get('num_layers', 1)
            test_every  = params.get('test_every', 1)
            bidirectional = params.get('bidirectional', False)


            # Split dataset
            # Prepare dataset
            params = cfg.parameters
            pairs  = cfg.currency_pairs
            seq_len = params.get('seq_len', 6)
            dataset = PSQLDataset(app, pairs, seq_len=seq_len)
            if len(dataset) < seq_len:
                raise RuntimeError("Insufficient data length.")

            # Correct sequential split
            n_total = len(dataset)
            n_train = int(0.8 * n_total)  # 20% for training

            # Create explicit subsets
            train_ds = torch.utils.data.Subset(dataset, indices=list(range(0, n_train)))
            test_ds  = torch.utils.data.Subset(dataset, indices=list(range(n_train, n_total)))

            # DataLoaders
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)   # Shuffle training
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)  # No shuffle testing

            # Model setup
            sample = dataset[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMVAE(
                input_size=sample.shape[1],
                hidden_size=hidden_size,
                latent_dim=latent_dim,
                num_layers=num_layers,
                bidirectional=bidirectional
            ).to(device)

            print(summary(model, input_size=(batch_size, seq_len, sample.shape[1])))

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training loop
            print(f"🚀 VAE training start config={config_id}, run={run.id}, version={run.version}")
            for epoch in range(1, epochs + 1):
                model.train()
                sum_loss = sum_kl = 0.0
                weight = base_kl * min(1.0, epoch / epochs)
                #weight = 0
                for batch in train_loader:
                    data = batch.to(device)
                    recon, mu, logvar = model(data)
                    loss = vae_loss(recon, data, mu, logvar, weight)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    sum_kl   += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)).item()
                avg_loss = sum_loss / len(train_loader)
                avg_kl   = sum_kl   / len(train_loader)
                session.add_all([
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train',    value=avg_loss),
                    ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='train_kl', value=avg_kl)
                ])

                if epoch % test_every == 0:
                    model.eval()
                    test_loss = test_kl = 0.0
                    with torch.no_grad():
                        for batch in test_loader:
                            data = batch.to(device)
                            recon, mu, logvar = model(data)
                            test_loss += nn.MSELoss()(recon, data).item()
                            test_kl   += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)).item()
                    session.add_all([
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test',    value=test_loss/len(test_loader)),
                        ModelLossHistory(model_run_id=run.id, epoch=epoch, loss_type='test_kl', value=test_kl  /len(test_loader))
                    ])
                    print(f"Epoch {epoch:03d}: Train Loss = {avg_loss:.4f} | Train KL = {avg_kl:.4f}")
            
            # Save model blob
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            run.model_blob = buffer.getvalue()

            # Save latent points and inputs
            print("💾 Saving latent points and inputs...")
            dates = dataset.get_common_dates()
            for i in range(len(dataset)):
                x = dataset[i].unsqueeze(0).to(device)
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar).squeeze(0).tolist()
                lp = LatentPoint(model_run_id=run.id, start_date=dates[i], lag=seq_len-1, latent_vector=z)
                session.add(lp)
                session.flush()
                #for pair in dataset.get_pair_codes():
                #    for d in dates[i:i+seq_len]:
                #        session.add(LatentInput(latent_point_id=lp.id, currency_pair=pair, date=d, source_table='exchange_rate'))

            # Commit
            session.commit()
            print(f"✅ Completed VAE run {run.id} for config {config_id}")
            return run.id
        except Exception as e:
            session.rollback()
            print(f"❌ Training failed for config {config_id}: {e}")
            raise
