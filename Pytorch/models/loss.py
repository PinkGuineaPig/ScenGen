# === Loss Function ===
import torch.nn as nn
import torch

def vae_loss(recon_x, x, mu, logvar, kl_weight=0.1):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_weight * kl_div