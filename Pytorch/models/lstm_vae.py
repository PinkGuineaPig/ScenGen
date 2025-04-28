import torch
import torch.nn as nn

# class LSTMVAE(nn.Module):
#     def __init__(self, input_size, hidden_size, latent_dim, num_layers=1, bidirectional=False):
#         super().__init__()
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.direction_factor = 2 if bidirectional else 1

#         # Encoder
#         self.encoder_lstm = nn.LSTM(
#             input_size, hidden_size,
#             batch_first=True,
#             num_layers=num_layers,
#             bidirectional=bidirectional
#         )

#         self.encoder_fc = nn.Sequential(
#             nn.Linear(hidden_size * self.direction_factor, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )

#         self.fc_mu = nn.Linear(hidden_size, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_size, latent_dim)

#         # Decoder
#         self.decoder_input = nn.Linear(latent_dim, hidden_size)
#         self.decoder_lstm = nn.LSTM(
#             hidden_size, hidden_size,
#             batch_first=True,
#             num_layers=num_layers,
#             bidirectional=False
#         )

#         # Output projection
#         self.output_proj = nn.Linear(hidden_size, input_size)

#     def encode(self, x):
#         _, (h_n, _) = self.encoder_lstm(x)
#         if self.bidirectional:
#             fw = h_n[-2]
#             bw = h_n[-1]
#             h = torch.cat((fw, bw), dim=1)
#         else:
#             h = h_n[-1]
#         h = self.encoder_fc(h)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z, seq_len):
#         if z.dim() == 1:
#             z = z.unsqueeze(0)
#         hidden = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
#         output, _ = self.decoder_lstm(hidden)
#         output = self.output_proj(output)  # << NEW line
#         return output

#     def forward(self, x):
#         seq_len = x.size(1)
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z, seq_len)
#         return recon, mu, logvar




import torch
import torch.nn as nn

class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1, bidirectional=False, max_seq_len=500):
        super().__init__()
        self.bidirectional = bidirectional

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Temporal embedding for decoding
        self.temporal_emb = nn.Embedding(max_seq_len, hidden_size)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        if self.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]
        h = self.encoder_fc(h_n)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        # Build time embeddings
        positions = torch.arange(seq_len, device=z.device)
        time_emb = self.temporal_emb(positions)                  # [seq_len, hidden_size]
        time_emb = time_emb.unsqueeze(0).repeat(z.size(0), 1, 1)  # [batch, seq_len, hidden_size]
        # Combine latent vector with time embeddings
        latent_rep = self.decoder_input(z).unsqueeze(1)          # [batch, 1, hidden_size]
        conditioned = latent_rep + time_emb                      # [batch, seq_len, hidden_size]

        outputs, _ = self.decoder_lstm(conditioned)
        return self.output_layer(outputs)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar