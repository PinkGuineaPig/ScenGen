import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
import math
from sqlalchemy import create_engine
from Backend.app import create_app, db
from Backend.app.models.data_models import Currency, ExchangeRate
from sqlalchemy.orm import sessionmaker

# --- Configuration ---
BATCH_SIZE = 128
TARGET_BATCH = 64000  # Valid batch (sequences 8192000 to 8192128)
MODEL_PATH = "logs/nan_cfg6_ep4_batch85342.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DB_URI = "postgresql://scengen_dev_user:dev_secret@localhost/scengen_dev"
SEQ_LEN = 500
STRIDE = 1
FFILL_LIMIT = 2

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Dataset Class ---
class ForexDataset(Dataset):
    def __init__(self, data, start_seq_idx, num_sequences):
        self.data = data.astype(np.float32)
        self.start_seq_idx = start_seq_idx
        self.num_sequences = num_sequences
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        seq_idx = self.start_seq_idx + idx * STRIDE
        start_idx = seq_idx
        end_idx = start_idx + SEQ_LEN
        return torch.tensor(self.data[start_idx:end_idx], dtype=torch.float32)

# --- LSTMVAE Model ---
class LSTMVAE(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, latent_dim=8, num_layers=2, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Positional encoding
        pe = torch.zeros(5000, input_size)
        for pos in range(5000):
            for i in range(0, input_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / input_size)))
                if i + 1 < input_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / input_size)))
        self.register_buffer('pe', pe)

        # Conv1d layer before encoder
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.conv_bn = nn.BatchNorm1d(input_size)
        self.conv_relu = nn.ReLU()

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        enc_out_size = hidden_size * self.num_directions
        self.encoder_fc = nn.Sequential(
            nn.Linear(enc_out_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, input_size)
        self.decoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.decoder_fc = nn.Linear(hidden_size * self.num_directions, input_size)

    def add_pe(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len].unsqueeze(0).expand(x.size(0), seq_len, self.input_size)
        return x + pe

    def encode(self, x):
        x = self.add_pe(x)
        logger.debug(f"After PE: min={x.min().item():.4f}, max={x.max().item():.4f}")
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        logger.debug(f"After conv1d: min={x.min().item():.4f}, max={x.max().item():.4f}")
        x = self.conv_bn(x)
        logger.debug(f"After conv_bn: min={x.min().item():.4f}, max={x.max().item():.4f}")
        x = self.conv_relu(x)
        logger.debug(f"After conv_relu: min={x.min().item():.4f}, max={x.max().item():.4f}")
        x = x.permute(0, 2, 1)
        _, (h, _) = self.encoder(x)
        h = h[-1] if not self.bidirectional else torch.cat((h[-2], h[-1]), dim=-1)
        logger.debug(f"After encoder: min={h.min().item():.4f}, max={h.max().item():.4f}")
        h = self.encoder_fc(h)
        logger.debug(f"After encoder_fc: min={h.min().item():.4f}, max={h.max().item():.4f}")
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -20, 20)
        logger.debug(f"Mu: min={mu.min().item():.4f}, max={mu.max().item():.4f}")
        logger.debug(f"Logvar: min={logvar.min().item():.4f}, max={logvar.max().item():.4f}")
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(min=-20, max=20))
        eps = torch.randn_like(std)
        z = mu + eps * std
        logger.debug(f"Z: min={z.min().item():.4f}, max={z.max().item():.4f}")
        return z

    def decode(self, z, seq_len=500):
        h = self.fc_dec(z).unsqueeze(1).repeat(1, seq_len, 1)
        logger.debug(f"After fc_dec: min={h.min().item():.4f}, max={h.max().item():.4f}")
        x, _ = self.decoder(h)
        logger.debug(f"After decoder: min={x.min().item():.4f}, max={x.max().item():.4f}")
        x = self.decoder_fc(x)
        logger.debug(f"After decoder_fc: min={x.min().item():.4f}, max={x.max().item():.4f}")
        return x

    def forward(self, x):
        if not self.training:
            self.encoder_fc.eval()
            self.conv_bn.eval()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, seq_len=x.size(1)), mu, logvar

# --- Load and Preprocess Data from Database ---
def load_data_from_db():
    logger.info("Connecting to database...")
    app = create_app({"SQLALCHEMY_DATABASE_URI": DB_URI})
    
    with app.app_context():
        try:
            eur = db.session.query(Currency).filter_by(code='EUR').first()
            aud = db.session.query(Currency).filter_by(code='AUD').first()
            if not eur or not aud:
                raise ValueError(f"EUR (id={eur.id if eur else None}) or AUD (id={aud.id if aud else None}) not found")
            if eur.id != 1 or aud.id != 4:
                logger.warning(f"Expected EUR.id=1, AUD.id=4; got EUR.id={eur.id}, AUD.id={aud.id}")
            
            total_records = db.session.query(ExchangeRate).filter(
                ExchangeRate.base_currency_id == eur.id,
                ExchangeRate.quote_currency_id == aud.id
            ).count()
            logger.info(f"Total EUR/AUD records: {total_records}")
            
            first_rate = db.session.query(ExchangeRate).filter(
                ExchangeRate.base_currency_id == eur.id,
                ExchangeRate.quote_currency_id == aud.id
            ).order_by(ExchangeRate.timestamp).first()
            last_rate = db.session.query(ExchangeRate).filter(
                ExchangeRate.base_currency_id == eur.id,
                ExchangeRate.quote_currency_id == aud.id
            ).order_by(ExchangeRate.timestamp.desc()).first()
            logger.info(f"Timestamp range: {first_rate.timestamp} to {last_rate.timestamp}")
            
            max_sequences = max(0, (total_records - SEQ_LEN) // STRIDE + 1)
            logger.info(f"Maximum possible sequences: {max_sequences}")
            
            target_seq_start = TARGET_BATCH * BATCH_SIZE
            target_seq_end = (TARGET_BATCH + 1) * BATCH_SIZE
            if target_seq_start >= max_sequences:
                raise ValueError(f"Target batch {TARGET_BATCH} (sequences {target_seq_start} to {target_seq_end}) exceeds max sequences {max_sequences}")
            
            buffer = 5000
            query_start = max(0, target_seq_start - buffer)
            query_end = min(total_records, target_seq_end + SEQ_LEN - 1 + buffer)
            num_records = query_end - query_start
            
            logger.info(f"Querying records {query_start} to {query_end} ({num_records} records)")
            rates = db.session.query(ExchangeRate).filter(
                ExchangeRate.base_currency_id == eur.id,
                ExchangeRate.quote_currency_id == aud.id
            ).order_by(ExchangeRate.timestamp).offset(query_start).limit(num_records).all()
            
            if not rates:
                raise ValueError(f"No EUR/AUD exchange rates found for records {query_start} to {query_end}")
            
            df = pd.DataFrame([
                {'timestamp': rate.timestamp, 'close': float(rate.close)} for rate in rates
            ])
            
            logger.info(f"Loaded {len(df)} EUR/AUD records")
            
            df[['close']] = df[['close']].ffill(limit=FFILL_LIMIT)
            
            if df[['close']].isna().any().any():
                logger.warning("NaN values detected after ffill")
                df[['close']] = df[['close']].fillna(0.0)
            if np.isinf(df[['close']].values).any():
                logger.warning("Inf values detected in data")
                df[['close']] = df[['close']].replace([np.inf, -np.inf], 0.0)
            
            data = df[['close']].values.astype(np.float32)
            data_mean = data.mean()
            data_std = data.std() + 1e-6
            data = (data - data_mean) / data_std
            data = np.clip(data * 5.0, -5.0, 5.0)
            data = np.repeat(data, 2, axis=1)
            
            logger.info(f"Data shape after preprocessing: {data.shape}")
            return data, query_start
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise

# --- Loss Functions ---
def compute_recon_loss(recon_x, x):
    recon_x = torch.nan_to_num(recon_x, nan=0.0, posinf=5.0, neginf=-5.0)
    return nn.functional.mse_loss(recon_x, x, reduction="sum") / x.size(0)

def compute_kl_loss(mu, logvar, kl_weight=0.5):
    mu = torch.nan_to_num(mu, nan=0.0)
    logvar = torch.clamp(logvar, min=-20, max=20)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().clamp(max=1e10))
    return kl_weight * kl / mu.size(0)

# --- Inspect Checkpoint Weights ---
def inspect_checkpoint(checkpoint):
    logger.info("Inspecting checkpoint weights...")
    for key, param in checkpoint.items():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"NaN or Inf detected in {key}")
            logger.warning(f"NaN count: {torch.isnan(param).sum().item()}")
            logger.warning(f"Inf count: {torch.isinf(param).sum().item()}")
        logger.debug(f"{key}: min={param.min().item():.4f}, max={param.max().item():.4f}")

# --- Inspect Batch ---
def inspect_batch(dataloader, target_batch, model):
    logger.info(f"Inspecting batch {target_batch}...")
    
    for i, data in enumerate(dataloader):
        if i == 0:
            data = data.to(DEVICE)
            
            logger.info("\nInput Data Statistics:")
            logger.info(f"Shape: {data.shape}")
            logger.info(f"Min: {data.min().item():.4f}, Max: {data.max().item():.4f}")
            logger.info(f"Mean: {data.mean().item():.4f}, Std: {data.std().item():.4f}")
            if torch.isnan(data).any() or torch.isinf(data).any():
                logger.warning("NaN or Inf detected in input data!")
                logger.warning(f"NaN count: {torch.isnan(data).sum().item()}")
                logger.warning(f"Inf count: {torch.isinf(data).sum().item()}")
            
            with torch.no_grad():
                recon_data, mu, logvar = model(data)
                
                logger.info("\nModel Output Statistics (Reconstruction):")
                logger.info(f"Shape: {recon_data.shape}")
                logger.info(f"Min: {recon_data.min().item():.4f}, Max: {recon_data.max().item():.4f}")
                logger.info(f"Mean: {recon_data.mean().item():.4f}, Std: {recon_data.std().item():.4f}")
                if torch.isnan(recon_data).any() or torch.isinf(recon_data).any():
                    logger.warning("NaN or Inf detected in reconstruction!")
                
                logger.info("\nLatent Variables Statistics:")
                logger.info(f"Mu - Min: {mu.min().item():.4f}, Max: {mu.max().item():.4f}, Mean: {mu.mean().item():.4f}")
                logger.info(f"Logvar - Min: {logvar.min().item():.4f}, Max: {logvar.max().item():.4f}, Mean: {logvar.mean().item():.4f}")
                if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(logvar).any() or torch.isinf(logvar).any():
                    logger.warning("NaN or Inf detected in latent variables!")
                
                recon_loss = compute_recon_loss(recon_data, data)
                kl_loss = compute_kl_loss(mu, logvar, kl_weight=0.5)
                total_loss = recon_loss + kl_loss
                
                logger.info("\nLoss Values:")
                logger.info(f"Reconstruction Loss: {recon_loss.item():.4f}")
                logger.info(f"KL Loss: {kl_loss.item():.4f}")
                logger.info(f"Total Loss: {total_loss.item():.4f}")
                if not torch.isfinite(total_loss):
                    logger.warning("NaN or Inf detected in loss!")
            
            torch.save(data.cpu(), f"batch_{target_batch}.pt")
            logger.info(f"Batch data saved to batch_{target_batch}.pt")
            break
    else:
        logger.error(f"Batch not found in dataloader. Total sequences: {len(dataloader)}")

# --- Main ---
def main():
    try:
        # Load data
        data, query_start = load_data_from_db()
        num_sequences = min(BATCH_SIZE, max(0, (len(data) - SEQ_LEN) // STRIDE + 1))
        dataset = ForexDataset(data, start_seq_idx=TARGET_BATCH * BATCH_SIZE - query_start, num_sequences=num_sequences)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = LSTMVAE(
            input_size=2,
            hidden_size=64,
            latent_dim=8,
            num_layers=2,
            bidirectional=False
        ).to(DEVICE)
        
        # Load and inspect checkpoint
        logger.info(f"Loading checkpoint from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        inspect_checkpoint(checkpoint)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        logger.info("Model loaded successfully")
        
        # Inspect batch
        with torch.autograd.set_detect_anomaly(True):
            inspect_batch(dataloader, target_batch=TARGET_BATCH, model=model)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main()