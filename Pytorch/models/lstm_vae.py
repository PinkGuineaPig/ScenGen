import torch
import torch.nn as nn
import math

class LSTMVAE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_dim: int, num_layers: int = 1,
                 bidirectional: bool = False, max_len: int = 1000, ema_alpha: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.ema_alpha = ema_alpha

        self.fc_input = nn.Linear(input_size, 512)
        self.encoder = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc_mu = nn.Linear(hidden_size * self.num_directions, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * self.num_directions, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc_out = nn.Linear(hidden_size * self.num_directions, input_size)

        self.pe = self._build_pe(max_len, 512).to(torch.device('cpu'))

    def ema(self, data: torch.Tensor, alpha: float = None) -> torch.Tensor:
        alpha = alpha or self.ema_alpha
        ema = data.clone()
        # Apply EMA only to squared returns (index 1)
        for t in range(1, data.size(1)):
            ema[:, t, 1] = alpha * data[:, t, 1] + (1 - alpha) * ema[:, t-1, 1]
        return ema

    def _build_pe(self, max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term) * 0.1
        pe[:, 1::2] = torch.cos(position * div_term) * 0.1
        return pe

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        x = torch.relu(self.fc_input(x))
        x = x + self.pe[:seq_len].unsqueeze(0).to(x.device)
        _, (h, _) = self.encoder(x)
        h = h[-self.num_directions:].transpose(0, 1).contiguous().view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = z.size(0)
        x = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        x, _ = self.decoder(x)
        x = self.fc_out(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.ema(x, alpha=self.ema_alpha)  # Apply EMA only to squared returns
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar