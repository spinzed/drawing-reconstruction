import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, ContinuousBernoulli, kl_divergence

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class GaussianEncoder(nn.Module):
    """Convolutional encoder producing (mu, logvar)"""
    def __init__(self, in_channels, latent_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # Automatically compute flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_dim, input_dim)
            h = self.net(dummy)
            self.flatten_dim = h.view(1, -1).size(1)

        self.mean = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        return self.mean(h), self.logvar(h)

class GaussianDecoder(nn.Module):
    """Decoder modeling p(x|z,y)"""
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.init_channels = 128
        self.init_spatial = input_dim // 4  # match encoder downsampling

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.init_channels * self.init_spatial * self.init_spatial),
            nn.LeakyReLU(0.2)
        )

        # Conv decoder
        self.trunk = nn.Sequential(
            nn.ConvTranspose2d(self.init_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.mean_head = nn.Conv2d(32, 1, 3, padding=1)  # output logits for Bernoulli

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.init_channels, self.init_spatial, self.init_spatial)

        h = self.trunk(h)
        x_logits = self.mean_head(h)
        return x_logits

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=200, device="cpu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.encoder_prior = GaussianEncoder(in_channels=1, latent_dim=latent_dim, input_dim=input_dim)
        self.encoder_post = GaussianEncoder(in_channels=2, latent_dim=latent_dim, input_dim=input_dim)

        self.decoder = GaussianDecoder(latent_dim=latent_dim, input_dim=input_dim)

    def forward(self, x, y):
        mu_p, logvar_p = self.encoder_prior(y)
        
        xy = torch.cat([x, y], dim=1)
        mu_q, logvar_q = self.encoder_post(xy)

        z = reparametrize(mu_q, logvar_q)

        x_logits = self.decoder(z)
        x_prob = torch.sigmoid(x_logits)

        return z, mu_p, logvar_p, mu_q, logvar_q, x_logits, x_prob

    def sample(self, y):
        mu_p, logvar_p = self.encoder_prior(y)
        #z = reparametrize(mu_p, logvar_p)        
        z = mu_p
        x_logits = self.decoder(z)
        return torch.sigmoid(x_logits)

    @staticmethod
    def loss(x, y, mu_p, logvar_p, mu_q, logvar_q, x_logits, beta=1.0):
        """CVAE ELBO loss"""
        recon_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none')
        recon_loss = recon_loss.flatten(1).mean(1)  

        q_dist = Normal(mu_q, torch.exp(0.5 * logvar_q))
        p_dist = Normal(mu_p, torch.exp(0.5 * logvar_p))
        kl = kl_divergence(q_dist, p_dist).mean(1)

        loss = (recon_loss + beta * kl).mean()
        return loss
